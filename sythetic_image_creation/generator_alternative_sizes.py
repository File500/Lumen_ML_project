import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
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
    Build the generator model for the GAN.
    Input: Random noise vector
    Output: Generated melanoma image (500x500x3)
    """
    noise_dim = 100

    model = models.Sequential()

    # Foundation for 8x8 image
    model.add(layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Reshape to 8x8x512
    model.add(layers.Reshape((8, 8, 512)))

    # Upsample to 16x16
    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 32x32
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 64x64
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 125x125
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 250x250
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample to 500x500
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # Upsample to 1000x1000
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())


    # Final output layer with tanh activation (pixels in [-1, 1])
    model.add(layers.Conv2D(3, (5, 5), padding='same', activation='tanh', use_bias=False))

    return model


def build_discriminator():
    """
    Build the discriminator model for the GAN.
    Input: Melanoma image (500x500x3)
    Output: Real/Fake probability
    """
    model = models.Sequential()

    # Input 500x500x3 image
    model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same', input_shape=[1000, 1000, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Downsample to 250x250
    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Downsample to 125x125
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Downsample to 64x64
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Downsample to 32x32
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Downsample to 16x16
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Downsample to 8x8
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Flatten and output binary probability
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


class MelanomaGAN:
    def __init__(self, data_dir, batch_size=16):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = (1000, 1000)  # Changed to 500x500
        self.channels = 3
        self.noise_dim = 100

        # Build and compile models
        self.generator = build_generator()
        self.discriminator = build_discriminator()

        # Define optimizers
        self.gen_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.disc_optimizer = optimizers.Adam(learning_rate=0.00001, beta_1=0.5)

        # Metrics for tracking progress
        self.gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
        self.disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')

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
            image_size=self.img_size,  # Now using 500x500
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
        ])

        # Apply augmentation
        self.dataset = self.dataset.map(
            lambda x: (data_augmentation(x, training=True))
        )

    def _discriminator_loss(self, real_output, fake_output):
        """Calculate the discriminator loss"""
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=real_output, labels=tf.ones_like(real_output)
            )
        )
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_output, labels=tf.zeros_like(fake_output)
            )
        )
        total_loss = real_loss + fake_loss
        return total_loss

    def _generator_loss(self, fake_output):
        """Calculate the generator loss"""
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_output, labels=tf.ones_like(fake_output)
            )
        )

    @tf.function
    def _train_step(self, images):
        """Perform one training step with a batch of images"""
        # Generate noise for the generator
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images
            generated_images = self.generator(noise, training=True)

            # Get discriminator outputs for real and fake images
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Calculate losses
            gen_loss = self._generator_loss(fake_output)
            disc_loss = self._discriminator_loss(real_output, fake_output)

        # Compute gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        # Update metrics
        self.gen_loss_metric.update_state(gen_loss)
        self.disc_loss_metric.update_state(disc_loss)

        return gen_loss, disc_loss

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

                gen_loss, disc_loss = self._train_step(images_batch)

                pbar.update(1)
                pbar.set_description(f"G Loss: {gen_loss:.4f}, D Loss: {disc_loss:.4f}")

            pbar.close()

            # Print epoch results
            print(f"Epoch {epoch}: Generator Loss = {self.gen_loss_metric.result():.4f}, "
                  f"Discriminator Loss = {self.disc_loss_metric.result():.4f}")

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
                fig, axs = plt.subplots(4, 4, figsize=(16, 16))  # Adjusted figure size for 500x500 images
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
    parser = argparse.ArgumentParser(description='Melanoma GAN for synthetic data generation')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing malignant melanoma images')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True,
                        help='Mode: train the model or generate synthetic images')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train (for train mode)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of synthetic images to generate (for generate mode)')
    parser.add_argument('--output_dir', type=str, default='generated_melanoma',
                        help='Directory to save generated images')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to pre-trained generator model to load (for generate mode, should end with .weights.h5)')

    args = parser.parse_args()

    # Initialize the GAN
    gan = MelanomaGAN(args.data_dir, batch_size=args.batch_size)

    if args.mode == 'train':
        # Train the model
        gan.train(epochs=args.epochs)
    elif args.mode == 'generate':
        if args.load_model:
            # Load pre-trained model
            gan.load_model(args.load_model)

        # Generate synthetic images
        gan.generate_images(num_images=args.num_images, output_dir=args.output_dir)


if __name__ == '__main__':
    main()