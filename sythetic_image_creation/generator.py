import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.losses import BinaryCrossentropy
import keras
from keras.utils import image_dataset_from_directory
from keras.preprocessing.image import img_to_array, array_to_img
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# GPU memory growth to prevent OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def build_generator():
    """
    Build the generator model with improved architecture.
    Input: Random noise vector
    Output: Generated melanoma image (224x224x3)
    """
    noise_dim = 128  # Increased noise dimension

    model = models.Sequential()

    # Foundation for 7x7 image
    model.add(layers.Dense(7 * 7 * 512, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Reshape to 7x7x512
    model.add(layers.Reshape((7, 7, 512)))

    # Upsample to 14x14
    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Upsample to 28x28
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Upsample to 56x56
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Upsample to 112x112
    model.add(layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Upsample to 224x224
    model.add(layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Add residual block for preserving details
    model.add(ResidualBlock(16))

    # Final output layer with tanh activation (pixels in [-1, 1])
    model.add(layers.Conv2D(3, (4, 4), padding='same', activation='tanh', use_bias=False))

    return model


def build_discriminator():
    """
    Build the discriminator model with improved architecture.
    Input: Melanoma image (224x224x3)
    Output: Real/Fake probability
    """
    input_img = layers.Input(shape=(224, 224, 3))

    # Block 1: 224x224x3 -> 112x112x64
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(input_img)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 2: 112x112x64 -> 56x56x128
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 3: 56x56x128 -> 28x28x256
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 4: 28x28x256 -> 14x14x512
    x = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 5: 14x14x512 -> 7x7x1024
    x = layers.Conv2D(1024, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.3)(x)

    # Output block
    x = layers.Flatten()(x)
    output = layers.Dense(1)(x)

    model = models.Model(input_img, output)
    return model


# Self-attention module for GANs
class SelfAttention(layers.Layer):
    def __init__(self, channels, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = channels
        self.query_conv = layers.Conv2D(channels // 8, kernel_size=1)
        self.key_conv = layers.Conv2D(channels // 8, kernel_size=1)
        self.value_conv = layers.Conv2D(channels, kernel_size=1)
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)

    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        # Use tf.shape instead of .shape for dynamic batch size
        input_shape = tf.shape(inputs)
        batch_size, height, width = input_shape[0], input_shape[1], input_shape[2]

        # Linear projections
        f = self.query_conv(inputs)  # [B, H, W, C//8]
        g = self.key_conv(inputs)  # [B, H, W, C//8]
        h = self.value_conv(inputs)  # [B, H, W, C]

        # Get shapes dynamically
        f_channels = tf.shape(f)[-1]
        g_channels = tf.shape(g)[-1]
        h_channels = tf.shape(h)[-1]

        # Reshape for matrix multiplication - use tf.shape for dynamic sizing
        f_flatten = tf.reshape(f, [batch_size, height * width, f_channels])  # [B, H*W, C//8]
        g_flatten = tf.reshape(g, [batch_size, height * width, g_channels])  # [B, H*W, C//8]
        h_flatten = tf.reshape(h, [batch_size, height * width, h_channels])  # [B, H*W, C]

        # Compute attention map
        s = tf.matmul(f_flatten, g_flatten, transpose_b=True)  # [B, H*W, H*W]
        beta = tf.nn.softmax(s)  # Attention weights

        # Weight value by attention
        o = tf.matmul(beta, h_flatten)  # [B, H*W, C]
        o = tf.reshape(o, [batch_size, height, width, channels])  # [B, H, W, C]

        # Combine with input using learnable gamma
        return self.gamma * o + inputs


# Residual block for better detail preservation
class ResidualBlock(layers.Layer):
    def __init__(self, channels, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(channels, kernel_size=3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(channels, kernel_size=3, padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual  # Skip connection
        return tf.nn.leaky_relu(x, alpha=0.2)


class PerceptualLoss:
    def __init__(self):
        # Load VGG model as feature extractor
        vgg = applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        vgg.trainable = False
        for layer in vgg.layers:
            layer.trainable = False

        # Define layers from which to extract features
        self.layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
        self.layers = [vgg.get_layer(name).output for name in self.layer_names]

        # Create model
        self.model = models.Model(inputs=vgg.input, outputs=self.layers)

    def __call__(self, real_images, generated_images):
        # Convert from [-1, 1] to [0, 255] and preprocess for VGG
        real_images = (real_images + 1) * 127.5
        generated_images = (generated_images + 1) * 127.5

        # Preprocess using VGG preprocessing
        real_images = applications.vgg16.preprocess_input(real_images)
        generated_images = applications.vgg16.preprocess_input(generated_images)

        # Extract features
        real_features = self.model(real_images)
        gen_features = self.model(generated_images)

        # Calculate L1 loss between feature maps
        loss = 0
        for real_feat, gen_feat in zip(real_features, gen_features):
            loss += tf.reduce_mean(tf.abs(real_feat - gen_feat))

        return loss


class MelanomaGAN:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = (224, 224)
        self.channels = 3
        self.noise_dim = 128

        # Build and compile models
        self.generator = build_generator()
        self.discriminator = build_discriminator()

        # Define optimizers with lower learning rates for stability
        self.gen_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
        self.disc_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

        # Metrics for tracking progress
        self.gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
        self.disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')
        self.perceptual_loss_metric = tf.keras.metrics.Mean(name='perceptual_loss')

        # Initialize perceptual loss
        self.perceptual_loss = PerceptualLoss()

        # Create output directory for saving samples
        self.output_dir = 'generated_samples'
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup data loader
        self._setup_data_loader()

    def _setup_data_loader(self):
        """Set up the data generator to load melanoma images"""
        # Using the newer TF/Keras API for dataset creation
        self.dataset = image_dataset_from_directory(
            self.data_dir,
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=True,
            label_mode=None  # Only images, no labels
        )

        # Preprocessing function to scale images to [-1, 1]
        def preprocess(images):
            # Convert to float and scale to [-1, 1]
            images = tf.cast(images, tf.float32) / 127.5 - 1
            return images

        # Apply preprocessing
        self.dataset = self.dataset.map(preprocess)

        # Add augmentation
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomTranslation(0.2, 0.2),
            # Add color jitter for more diversity
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
        ])

        # Apply augmentation
        self.dataset = self.dataset.map(
            lambda x: (data_augmentation(x, training=True))
        )

        # Prefetch for better performance
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)

    def _wasserstein_loss(self, y_true, y_pred):
        """
        Wasserstein loss for improved GAN training stability
        """
        return tf.reduce_mean(y_true * y_pred)

    def _generator_loss(self, fake_output):
        """Calculate the generator loss using Wasserstein loss + perceptual loss"""
        # Wasserstein loss wants to maximize the fake output for the generator
        # So we pass ones_like here for y_true
        return -tf.reduce_mean(fake_output)

    def _discriminator_loss(self, real_output, fake_output):
        """Calculate the discriminator loss using Wasserstein loss"""
        # Real samples should output positive values
        real_loss = -tf.reduce_mean(real_output)
        # Fake samples should output negative values
        fake_loss = tf.reduce_mean(fake_output)

        return real_loss + fake_loss

    def gradient_penalty(self, real_images, fake_images):
        """
        Calculate gradient penalty for improved WGAN training
        """
        batch_size = tf.shape(real_images)[0]

        # Generate random interpolation factors
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)

        # Create interpolated images between real and fake
        interpolated = real_images + alpha * (fake_images - real_images)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # Get the discriminator output for interpolated images
            pred = self.discriminator(interpolated, training=True)

        # Calculate gradients of discriminator prediction with respect to interpolated images
        gradients = tape.gradient(pred, interpolated)

        # Calculate the norm of the gradients
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]) + 1e-8)

        # Calculate gradient penalty
        gradient_penalty = tf.reduce_mean(tf.square(gradients_norm - 1.0))

        return gradient_penalty

    @tf.function
    def _train_step(self, images):
        """Perform one training step with a batch of images"""
        # Generate noise for the generator
        noise = tf.random.normal([tf.shape(images)[0], self.noise_dim])

        # Train discriminator
        with tf.GradientTape() as disc_tape:
            # Generate fake images
            generated_images = self.generator(noise, training=True)

            # Get discriminator outputs for real and fake images
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Calculate discriminator loss
            disc_loss = self._discriminator_loss(real_output, fake_output)

            # Add gradient penalty for WGAN-GP
            gp = self.gradient_penalty(images, generated_images)
            disc_loss += 10.0 * gp  # Lambda = 10 for GP

        # Compute gradients for discriminator and apply them
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        # Train generator (typically multiple steps for each discriminator step)
        for _ in range(2):  # Train generator more often for better quality
            with tf.GradientTape() as gen_tape:
                # Generate fake images
                noise = tf.random.normal([tf.shape(images)[0], self.noise_dim])
                generated_images = self.generator(noise, training=True)

                # Get discriminator output for fake images
                fake_output = self.discriminator(generated_images, training=True)

                # Calculate generator loss
                gen_loss = self._generator_loss(fake_output)

                # Add perceptual loss for enhanced details
                perceptual_loss = self.perceptual_loss(images, generated_images)
                gen_loss += 0.1 * perceptual_loss  # Weighted perceptual loss

            # Compute gradients for generator and apply them
            gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

            # Update metrics
            self.gen_loss_metric.update_state(gen_loss)
            self.disc_loss_metric.update_state(disc_loss)
            self.perceptual_loss_metric.update_state(perceptual_loss)

        return gen_loss, disc_loss, perceptual_loss

    def generate_images(self, num_images, output_dir=None):
        """Generate a specified number of melanoma images and save them"""
        if output_dir is None:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Generate images in batches to avoid memory issues
        batch_size = min(32, num_images)
        num_batches = (num_images + batch_size - 1) // batch_size  # ceiling division

        print(f"Generating {num_images} synthetic melanoma images...")

        count = 0
        for i in range(num_batches):
            # Calculate how many images to generate in this batch
            current_batch_size = min(batch_size, num_images - count)

            # Generate noise and create images
            noise = tf.random.normal([current_batch_size, self.noise_dim])
            generated_images = self.generator(noise, training=False)

            # Rescale from [-1, 1] to [0, 255] for saving as images
            generated_images = (generated_images * 127.5 + 127.5).numpy().astype(np.uint8)

            # Save each image in the batch
            for j in range(current_batch_size):
                count += 1
                img_path = os.path.join(output_dir, f'synthetic_melanoma_{count:04d}.png')
                plt.imsave(img_path, generated_images[j])

                if count % 10 == 0 or count == num_images:
                    print(f"Generated {count}/{num_images} images")

            # Optional: Add some image enhancement for even more clarity
            # This would require additional libraries like PIL or OpenCV

        print(f"Finished generating {num_images} synthetic melanoma images in {output_dir}")

    def train(self, epochs, save_interval=10):
        """Train the GAN for a specified number of epochs"""
        # Get dataset size
        dataset_size = tf.data.experimental.cardinality(self.dataset).numpy()
        if dataset_size == tf.data.experimental.UNKNOWN_CARDINALITY:
            print("Dataset size unknown, estimating based on first epoch...")
            steps_per_epoch = None
        else:
            steps_per_epoch = dataset_size

        print(f"Starting training for {epochs} epochs...")

        # Create directory for saving samples during training
        sample_dir = os.path.join(self.output_dir, 'training_samples')
        os.makedirs(sample_dir, exist_ok=True)

        # Create directory for saving models
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")

            # Reset metrics
            self.gen_loss_metric.reset_state()
            self.disc_loss_metric.reset_state()
            self.perceptual_loss_metric.reset_state()

            # Train for one epoch
            if steps_per_epoch:
                pbar = tqdm(total=steps_per_epoch)
            else:
                pbar = tqdm()

            # Iterate through dataset
            for images_batch in self.dataset:
                # Handle case where batch size might be smaller at the end
                if images_batch.shape[0] != self.batch_size:
                    # Pad the batch to the expected size
                    padding = self.batch_size - images_batch.shape[0]
                    # Repeat the first image to pad the batch
                    padding_images = tf.repeat(images_batch[0:1], repeats=padding, axis=0)
                    images_batch = tf.concat([images_batch, padding_images], axis=0)

                gen_loss, disc_loss, perceptual_loss = self._train_step(images_batch)

                pbar.update(1)
                pbar.set_description(
                    f"G Loss: {gen_loss:.4f}, D Loss: {disc_loss:.4f}, "
                    f"P Loss: {perceptual_loss:.4f}"
                )

            pbar.close()

            # Print epoch results
            print(f"Epoch {epoch}: Generator Loss = {self.gen_loss_metric.result():.4f}, "
                  f"Discriminator Loss = {self.disc_loss_metric.result():.4f}, "
                  f"Perceptual Loss = {self.perceptual_loss_metric.result():.4f}")

            # Generate and save sample images
            if epoch % save_interval == 0 or epoch == epochs:
                # Save model weights
                self.generator.save_weights(os.path.join(model_dir, f'generator_epoch_{epoch}.weights.h5'))
                self.discriminator.save_weights(os.path.join(model_dir, f'discriminator_epoch_{epoch}.weights.h5'))

                # Generate sample images
                print(f"Generating sample images for epoch {epoch}...")
                noise = tf.random.normal([16, self.noise_dim])
                generated_images = self.generator(noise, training=False)

                # Rescale images from [-1, 1] to [0, 255]
                generated_images = (generated_images * 127.5 + 127.5).numpy().astype(np.uint8)

                # Create a grid of sample images
                fig, axs = plt.subplots(4, 4, figsize=(10, 10))
                for i in range(4):
                    for j in range(4):
                        axs[i, j].imshow(generated_images[i * 4 + j])
                        axs[i, j].axis('off')

                plt.savefig(os.path.join(sample_dir, f'samples_epoch_{epoch}.png'))
                plt.close()

        print("Training completed!")

    def load_model(self, generator_path, discriminator_path=None):
        """Load pre-trained model weights"""
        # Make sure the path ends with .weights.h5
        if not generator_path.endswith('.weights.h5'):
            generator_path = generator_path.replace('.h5', '.weights.h5')

        self.generator.load_weights(generator_path)

        if discriminator_path:
            if not discriminator_path.endswith('.weights.h5'):
                discriminator_path = discriminator_path.replace('.h5', '.weights.h5')
            self.discriminator.load_weights(discriminator_path)

        print("Model loaded successfully")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Melanoma GAN for synthetic data generation')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing malignant melanoma images')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True,
                        help='Mode: train the model or generate synthetic images')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train (for train mode)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (larger batch size helps stability)')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of synthetic images to generate (for generate mode)')
    parser.add_argument('--output_dir', type=str, default='generated_melanoma',
                        help='Directory to save generated images')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to pre-trained generator model to load (should end with .weights.h5)')
    parser.add_argument('--load_disc_model', type=str, default=None,
                        help='Path to pre-trained discriminator model to load (should end with .weights.h5)')
    parser.add_argument('--continue_training', action='store_true',
                        help='Continue training from a previously saved model')

    args = parser.parse_args()

    # Initialize the GAN
    gan = MelanomaGAN(args.data_dir, batch_size=args.batch_size)

    if args.mode == 'train':
        if args.continue_training and args.load_model:
            # Load pre-trained models for continuing training
            print(f"Loading generator model from {args.load_model} to continue training")
            if args.load_disc_model:
                print(f"Loading discriminator model from {args.load_disc_model} to continue training")
                gan.load_model(args.load_model, args.load_disc_model)
            else:
                # Only load generator if discriminator isn't specified
                gan.load_model(args.load_model)

        # Train the model
        gan.train(epochs=args.epochs)
    elif args.mode == 'generate':
        if args.load_model:
            # Load pre-trained model
            gan.load_model(args.load_model)
        else:
            print("Error: Must provide a pre-trained model path with --load_model for generation mode")
            return

        # Generate synthetic images
        gan.generate_images(num_images=args.num_images, output_dir=args.output_dir)


if __name__ == '__main__':
    main()