#!/usr/bin/env python

from abc import ABC

import tensorflow as tf


class Model(ABC):
  def __init__(self, config):
    self._config = config

  @property
  def gen_learning(self):
    return 5e-5

  @property
  def disc_learning(self):
    return 5e-5

  def get_generator(self):
    return self.Generator(self._config) # pylint: disable=no-member

  def get_discriminator(self):
    return self.Discriminator(self._config) # pylint: disable=no-member

  all_individual_losses = ["adversarial", "binary", "ground_truth",
      "relevancy", "reconstruction", "patho_reconstruction",
      "variation", "identity"]

  @property
  def individual_loss_fields(self):
    # this returns only the actually used individual losses
    arguments = vars(self._config)
    losses = []
    for loss in Model.all_individual_losses:
      if arguments["loss_{}".format(loss)]:
        losses.append(loss)
    return losses

  def gen_loss(self, disc_on_generated, loss_args):
    """Loss for the generator. Intended to be overwritten when desired.

    :disc_on_generated: Predictions on generated data.
    :loss_args: Additional arguments for loss calculation.
    :returns: A dictionary with the losses where the value is a tuple with the weighted and unweighted loss.
    """

    # NOTE: could make this dynamic too but that'd require all losses to have the same signature
    losses = {}

    if self._config.loss_adversarial:
      loss = Model._gen_adversarial_loss(disc_on_generated)
      weight = self._config.loss_adversarial
      losses["adversarial"] = ((loss+1) ** self._config.loss_adversarial_power - 1) * weight
    if self._config.loss_binary:
      loss = Model._gen_binary_loss(loss_args.generated_images)
      weight = self._config.loss_binary
      losses["binary"] = ((loss+1) ** self._config.loss_binary_power - 1) * weight
    if self._config.loss_ground_truth:
      loss = Model._gen_ground_truth_loss(loss_args.generated_images, loss_args.targets)
      weight = self._config.loss_ground_truth
      losses["ground_truth"] = ((loss+1) ** self._config.loss_ground_truth_power - 1) * weight
    if self._config.loss_relevancy:
      loss = Model._gen_relevancy_loss(loss_args.generated_images, loss_args.inputs)
      weight = self._config.loss_relevancy
      losses["relevancy"] = ((loss+1) ** self._config.loss_relevancy_power - 1) * weight
    if self._config.loss_reconstruction:
      loss = Model._gen_reconstruction_loss(loss_args.inputs, loss_args.reconstructed_images)
      weight = self._config.loss_reconstruction
      losses["reconstruction"] = ((loss+1) ** self._config.loss_reconstruction_power - 1) * weight
    if self._config.loss_patho_reconstruction:
      loss = Model._gen_patho_reconstruction_loss(loss_args.inputs, loss_args.reconstructed_images)
      weight = self._config.loss_patho_reconstruction
      losses["patho_reconstruction"] = ((loss+1) ** self._config.loss_patho_reconstruction_power - 1) * weight
    if self._config.loss_variation:
      loss = Model._gen_variation_loss(loss_args.generated_images)
      weight = self._config.loss_variation
      losses["variation"] = ((loss+1) ** self._config.loss_variation_power - 1) * weight
    if self._config.loss_identity:
      loss = Model._gen_identity_loss(loss_args.targets, loss_args.identity_images)
      weight = self._config.loss_identity
      losses["identity"] = ((loss+1) ** self._config.loss_identity_power - 1) * weight

    return losses

  @staticmethod
  def disc_loss(disc_on_real, disc_on_generated):
    """Loss for the discriminator. Intended to be overwritten when desired.

    :disc_on_real: Predictions on real data.
    :disc_on_generated: Predictions on generated data.
    :returns: Loss for the discriminator.

    """
    return Model._disc_adversarial_loss(disc_on_real, disc_on_generated)

  @staticmethod
  def disc_k_plus_one_class_loss(disc_on_real, real_labels, disc_on_generated, generated_labels):
    """Loss for the discriminator when using a semi-supervised k+1 - class setting.

    :disc_on_real: Predictions on real data.
    :real_labels: The one-hot encoded labels of the real data.
    :disc_on_generated: Predictions on generated data.
    :generated_labels: The one-hot encoded labels of the generated data.
    :returns: Loss for the discriminator.

    """
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=real_labels, logits=disc_on_real)
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=generated_labels, logits=disc_on_generated)
    # real_loss = tf.losses.softmax_cross_entropy(onehot_labels=real_labels, logits=disc_on_real)
    # generated_loss = tf.losses.softmax_cross_entropy(onehot_labels=generated_labels, logits=disc_on_generated)
    return real_loss + generated_loss

  @staticmethod
  def _gen_adversarial_loss(disc_on_generated):
    """Adverarial loss for generator.

    :disc_on_generated: Predictions on generated data.
    :returns: Adversarial loss.

    """
    return tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(disc_on_generated),
        logits=disc_on_generated)

  @staticmethod
  def _gen_binary_loss(generated_images):
    """Loss for generated images that should be binary (-1/+1).

    :generated_images: The images that were generated and should be binary.
    :returns: Binary loss.

    """
    return tf.losses.mean_squared_error(tf.ones_like(generated_images), tf.abs(generated_images))

  @staticmethod
  def _gen_ground_truth_loss(generated_images, target_images):
    """Loss for generated images that should match some target images.

    :generated_images: The images that were generated and should match some target images.
    :target_images: The images that were the targets of the generation.
    :returns: Ground truth loss.

    """
    return tf.losses.mean_squared_error(target_images, generated_images)

  @staticmethod
  def _gen_relevancy_loss(generated_images, inputs):
    """Loss for generated images that should only differ in segmented regions from the original images.

    :generated_images: The images that were generated and should match some target images.
    :inputs: The inputs to the generator, consisting of image and segmentation.
    :returns: Relevancy loss.

    """
    assert inputs.shape[-1] == 4
    original_images = inputs[:, :, :, :-1]
    change_segmentations = inputs[:, :, :, -1:]
    no_patho_regions = change_segmentations <= 0

    return tf.losses.mean_squared_error(original_images, generated_images,
        weights=tf.cast(no_patho_regions, tf.float32))

  @staticmethod
  def _gen_reconstruction_loss(inputs, reconstructed_images):
    """Loss for reconstructed images that should match the input images.

    :inputs: The inputs to the generator, consisting of image and segmentation.
    :reconstructed_images: The images that were reconstructed from the generated images and should
      match the original images.
    :returns: Reconstruction loss.

    """
    original_images = inputs[:, :, :, :reconstructed_images.shape[-1]]
    return tf.losses.mean_squared_error(original_images, reconstructed_images)

  @staticmethod
  def _gen_patho_reconstruction_loss(inputs, reconstructed_images):
    """Loss for reconstructed images that should match the input images in the segmented regions.

    :inputs: The inputs to the generator, consisting of image and segmentation.
    :reconstructed_images: The images that were reconstructed from the generated images and should
      match the original images in the segmented regions.
    :returns: Reconstruction loss.

    """
    original_images = inputs[:, :, :, :-1]
    change_segmentations = inputs[:, :, :, -1:]
    patho_regions = change_segmentations > 0
    return tf.losses.mean_squared_error(original_images, reconstructed_images,
        weights=tf.cast(patho_regions, tf.float32))

  @staticmethod
  def _gen_variation_loss(generated_images):
    """Loss for the noisiness of the generated images per voxel by way of total variation. "By voxel" meaning
    that the variation is computed as the mean variation per pixel in the image and per color channel.

    :generated_images: The images that were generated.
    :returns: Variation loss.

    """
    return tf.reduce_sum(tf.image.total_variation(generated_images)) / \
        tf.cast(tf.reduce_prod(generated_images.shape), dtype=tf.float32)

  @staticmethod
  def _gen_identity_loss(targets, identity_images):
    """Loss for the difference between target images and their transformation, which shouldn't change
    anything since the targets are already in the target domain.

    :targets: Images in the target domain.
    :identity_images: The images that were generated based on the targets.
    :returns: Identity loss.

    """
    return tf.losses.mean_squared_error(targets[:, :, :, :3], identity_images)


  @staticmethod
  def _disc_adversarial_loss(disc_on_real, disc_on_generated):
    """Adversarial loss for discriminator.

    :disc_on_real: Predictions on real data.
    :disc_on_generated: Predictions on generated data.
    :returns: Adversarial loss.

    """
    # real images should be predicted with 1s
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(disc_on_real),
        logits=disc_on_real,
        label_smoothing=0.1) if disc_on_real is not None else 0
    # generated images should be predicted with 0s
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(disc_on_generated),
        logits=disc_on_generated) if disc_on_generated is not None else 0
    return real_loss + generated_loss


  def print_model_summary(self, generator, discriminator, gen_input):
    samples = generator(gen_input[:1], training=False)
    gen_summary = []
    generator.summary(print_fn=gen_summary.append)
    gen_summary = "\n".join(gen_summary)
    tf.logging.info("Generator model:\n{}".format(gen_summary))
    disc_input = tf.concat([gen_input[:1], samples[:1]], axis=-1) if self._config.conditioned_discriminator else samples[:1]
    _ = discriminator(disc_input, training=False)
    disc_summary = []
    discriminator.summary(print_fn=disc_summary.append)
    disc_summary = "\n".join(disc_summary)
    tf.logging.info("Discriminator model:\n{}".format(disc_summary))
