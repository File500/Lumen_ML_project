import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configure PyTorch to use memory more efficiently
if torch.cuda.is_available():
    # Enable memory efficient operations
    torch.backends.cudnn.benchmark = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Memory management function to clear cache
def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# Print memory stats for debugging
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


# We're no longer using these helper classes, since we've implemented the U-Net differently
# to avoid the dimension matching issues. These classes can be removed, but to keep
# the code backward compatible, we'll just leave them here as empty placeholder classes.
class UNetDown(nn.Module):
    """Placeholder for backward compatibility"""

    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()

    def forward(self, x):
        return x


class UNetUp(nn.Module):
    """Placeholder for backward compatibility"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()

    def forward(self, x, skip_input):
        return x


class Generator(nn.Module):
    """
    U-Net Generator for the GAN.
    Input: Random noise vector (100,)
    Output: Generated melanoma image (3, 512, 512)
    """

    def __init__(self, noise_dim=100, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        # Initial dense layer to project and reshape noise
        self.fc = nn.Linear(noise_dim, 8 * 8 * 512)

        # U-Net architecture
        # Encoder (downsampling path)
        self.down1 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )  # 8x8 -> 4x4

        self.down2 = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )  # 4x4 -> 2x2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5)
        )  # 2x2 -> 1x1

        # Decoder (upsampling path)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )  # 1x1 -> 2x2

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )  # 2x2 -> 4x4

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )  # 4x4 -> 8x8

        # Continue upsampling to reach 512x512
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(256 + 512, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )  # 8x8 -> 16x16

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )  # 16x16 -> 32x32

        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 32x32 -> 64x64

        self.up7 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )  # 64x64 -> 128x128

        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )  # 128x128 -> 256x256

        # Final upsampling and output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(8, output_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )  # 256x256 -> 512x512

    def forward(self, x):
        # Project and reshape noise
        x = self.fc(x)
        x0 = x.view(-1, 512, 8, 8)  # Initial 8x8 feature map

        # Encoder
        x1 = self.down1(x0)  # 4x4
        x2 = self.down2(x1)  # 2x2
        x3 = self.bottleneck(x2)  # 1x1

        # Decoder with skip connections
        d1 = self.up1(x3)  # 2x2
        d1_cat = torch.cat((d1, x2), 1)  # 2x2, concatenate with skip connection

        d2 = self.up2(d1_cat)  # 4x4
        d2_cat = torch.cat((d2, x1), 1)  # 4x4, concatenate with skip connection

        d3 = self.up3(d2_cat)  # 8x8
        d3_cat = torch.cat((d3, x0), 1)  # 8x8, concatenate with skip connection

        # Continue upsampling without skip connections
        d4 = self.up4(d3_cat)  # 16x16
        d5 = self.up5(d4)  # 32x32
        d6 = self.up6(d5)  # 64x64
        d7 = self.up7(d6)  # 128x128
        d8 = self.up8(d7)  # 256x256

        # Final output
        out = self.final(d8)  # 512x512

        return out


class Discriminator(nn.Module):
    """
    Discriminator network for the GAN based on the provided image.
    Input: Melanoma image (3, 512, 512)
    Output: Real/Fake probability and intermediate features for feature matching loss
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        # First layer - Conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Second layer - InstanceNorm
        self.instNorm = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Feature extraction layers for feature matching loss
        self.features = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Min-max GAN loss (final classifier)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling to get a single value
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.instNorm(x1)
        features = self.features(x2)
        output = self.classifier(features)

        return output.view(-1, 1), features  # Return both the classification result and features for loss


class MelanomaGAN:
    def __init__(self, data_dir, batch_size=8, img_size=512):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = (img_size, img_size)
        self.channels = 3
        self.noise_dim = 100

        # Build models
        self.generator = Generator(noise_dim=self.noise_dim, output_channels=self.channels).to(device)
        self.discriminator = Discriminator().to(device)

        # Initialize weights
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)

        # Define optimizers
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

        # Define loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()  # Added for U-Net reconstruction

        # Create output directory for saving samples
        self.output_dir = 'generated_samples'
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup data loader
        self._setup_data_loader()

    def _weights_init(self, m):
        """Initialize network weights using Xavier initialization"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def _setup_data_loader(self):
        """Set up the data loader to load melanoma images"""
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
        ])

        # Load dataset
        dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)

        # Create dataloader - reduce num_workers to save memory
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

    def feature_matching_loss(self, real_features, fake_features):
        """Compute feature matching loss between real and fake features"""
        return torch.mean(torch.abs(torch.mean(real_features, dim=0) - torch.mean(fake_features, dim=0)))

    def train(self, epochs, save_interval=10, accumulation_steps=2):
        """Train the GAN for a specified number of epochs"""
        print(f"Starting training for {epochs} epochs...")
        print(f"Initial GPU memory status:")
        print_gpu_memory()

        # Create directories for saving samples and models
        sample_dir = os.path.join(self.output_dir, 'training_samples')
        model_dir = 'models'
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Fixed noise for generating sample images
        fixed_noise = torch.randn(4, self.noise_dim, device=device)

        # Lists to keep track of losses
        g_losses = []
        d_losses = []
        reconstruction_losses = []
        feature_matching_losses = []

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")

            # Initialize epoch statistics
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_rec_loss = 0.0
            epoch_fm_loss = 0.0
            batches = 0

            # Progress bar
            pbar = tqdm(self.dataloader)

            # For gradient accumulation
            self.disc_optimizer.zero_grad()
            self.gen_optimizer.zero_grad()
            d_loss_accumulated = 0
            g_loss_accumulated = 0
            rec_loss_accumulated = 0
            fm_loss_accumulated = 0
            batch_count = 0

            for real_images, _ in pbar:
                batch_count += 1

                # Move images to device
                real_images = real_images.to(device)
                batch_size = real_images.size(0)

                # -----------------
                # Train Discriminator
                # -----------------
                # Real images
                real_labels = torch.ones(batch_size, 1, device=device)
                real_output, real_features = self.discriminator(real_images)
                d_loss_real = self.adversarial_loss(real_output, real_labels)

                # Generate fake images
                noise = torch.randn(batch_size, self.noise_dim, device=device)
                with torch.no_grad():
                    fake_images = self.generator(noise)  # Don't track gradients here

                fake_labels = torch.zeros(batch_size, 1, device=device)
                fake_output, fake_features_detached = self.discriminator(fake_images.detach())
                d_loss_fake = self.adversarial_loss(fake_output, fake_labels)

                # Total discriminator loss (min-max GAN loss)
                d_loss = (d_loss_real + d_loss_fake) / accumulation_steps
                d_loss.backward(retain_graph=True)
                d_loss_accumulated += d_loss.item() * accumulation_steps

                # -----------------
                # Train Generator
                # -----------------
                # Generate fake images again but track gradients this time
                noise = torch.randn(batch_size, self.noise_dim, device=device)
                fake_images = self.generator(noise)

                # Try to fool the discriminator
                fake_output, fake_features = self.discriminator(fake_images)
                g_loss = self.adversarial_loss(fake_output, real_labels)

                # Feature matching loss - compare real features with fake features
                fm_loss = self.feature_matching_loss(real_features, fake_features)

                # Reconstruction loss - since we're using a U-Net architecture,
                # we want to encourage generation of realistic melanoma images
                # For simplicity, we can use L1 regularization towards a target value
                # In a more advanced setup, this could be a cycle consistency loss
                # or paired reconstruction loss if such data was available
                rec_loss = self.reconstruction_loss(fake_images, torch.zeros_like(fake_images) + 0.5)

                # Combined generator loss with weights for each component
                # Adjust these weights to prioritize different aspects of generation
                combined_g_loss = (g_loss + 10.0 * fm_loss + 5.0 * rec_loss) / accumulation_steps
                combined_g_loss.backward()

                g_loss_accumulated += g_loss.item() * accumulation_steps
                fm_loss_accumulated += fm_loss.item() * accumulation_steps
                rec_loss_accumulated += rec_loss.item() * accumulation_steps

                # Step optimizers every accumulation_steps
                if batch_count % accumulation_steps == 0:
                    self.disc_optimizer.step()
                    self.gen_optimizer.step()
                    self.disc_optimizer.zero_grad()
                    self.gen_optimizer.zero_grad()

                    # Update progress bar
                    batches += 1
                    epoch_g_loss += g_loss_accumulated
                    epoch_d_loss += d_loss_accumulated
                    epoch_rec_loss += rec_loss_accumulated
                    epoch_fm_loss += fm_loss_accumulated

                    pbar.set_description(
                        f"G: {g_loss_accumulated:.3f}, D: {d_loss_accumulated:.3f}, "
                        f"FM: {fm_loss_accumulated:.3f}, Rec: {rec_loss_accumulated:.3f}"
                    )

                    # Reset accumulated losses
                    d_loss_accumulated = 0
                    g_loss_accumulated = 0
                    fm_loss_accumulated = 0
                    rec_loss_accumulated = 0

                # Clear cache periodically
                if batch_count % 10 == 0:
                    clear_cuda_cache()

            # Calculate average losses for the epoch
            if batches > 0:
                epoch_g_loss /= batches
                epoch_d_loss /= batches
                epoch_rec_loss /= batches
                epoch_fm_loss /= batches

                g_losses.append(epoch_g_loss)
                d_losses.append(epoch_d_loss)
                reconstruction_losses.append(epoch_rec_loss)
                feature_matching_losses.append(epoch_fm_loss)

            # Print epoch results
            print(f"Epoch {epoch}: Generator Loss = {epoch_g_loss:.4f}, Discriminator Loss = {epoch_d_loss:.4f}")
            print(f"Feature Matching Loss = {epoch_fm_loss:.4f}, Reconstruction Loss = {epoch_rec_loss:.4f}")
            print_gpu_memory()

            # Generate and save sample images
            if epoch % save_interval == 0 or epoch == epochs:
                # Save model weights
                torch.save(self.generator.state_dict(), os.path.join(model_dir, f'generator_epoch_{epoch}.pth'))
                torch.save(self.discriminator.state_dict(), os.path.join(model_dir, f'discriminator_epoch_{epoch}.pth'))

                # Generate sample images
                print(f"Generating sample images for epoch {epoch}...")
                with torch.no_grad():
                    self.generator.eval()
                    fake_images = self.generator(fixed_noise)
                    self.generator.train()

                # Save grid of images
                grid = vutils.make_grid(fake_images, padding=2, normalize=True, nrow=2)
                plt.figure(figsize=(10, 10))
                plt.axis("off")
                plt.title(f"Epoch {epoch}")
                plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
                plt.savefig(os.path.join(sample_dir, f'samples_epoch_{epoch}.png'))
                plt.close()

                # Clear cache after sample generation
                clear_cuda_cache()

        print("Training completed!")

        # Plot loss curves
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.title("Generator and Discriminator Losses")
        plt.plot(g_losses, label="Generator")
        plt.plot(d_losses, label="Discriminator")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.title("Feature Matching and Reconstruction Losses")
        plt.plot(feature_matching_losses, label="Feature Matching")
        plt.plot(reconstruction_losses, label="Reconstruction")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'loss_curves.png'))

    def generate_images(self, num_images, output_dir=None):
        """Generate a specified number of melanoma images and save them"""
        if output_dir is None:
            output_dir = self.output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Set generator to evaluation mode
        self.generator.eval()

        # Generate images in smaller batches to avoid memory issues
        batch_size = min(4, num_images)
        num_batches = (num_images + batch_size - 1) // batch_size  # ceiling division

        print(f"Generating {num_images} synthetic melanoma images...")
        print_gpu_memory()

        count = 0
        with torch.no_grad():
            for i in range(num_batches):
                # Calculate how many images to generate in this batch
                current_batch_size = min(batch_size, num_images - count)

                # Generate noise and create images
                noise = torch.randn(current_batch_size, self.noise_dim, device=device)
                fake_images = self.generator(noise)

                # Denormalize images from [-1, 1] to [0, 1]
                fake_images = (fake_images * 0.5) + 0.5

                # Save each image in the batch
                for j in range(current_batch_size):
                    count += 1
                    img_path = os.path.join(output_dir, f'synthetic_melanoma_{count:04d}.png')
                    vutils.save_image(fake_images[j], img_path)

                    if count % 10 == 0 or count == num_images:
                        print(f"Generated {count}/{num_images} images")

                # Clear cache after each batch
                clear_cuda_cache()
                if i % 5 == 0:
                    print_gpu_memory()

        # Set generator back to training mode
        self.generator.train()
        print(f"Finished generating {num_images} synthetic melanoma images in {output_dir}")

    def load_model(self, generator_path, discriminator_path=None):
        """Load pre-trained model weights"""
        self.generator.load_state_dict(torch.load(generator_path, map_location=device))

        if discriminator_path:
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))

        print("Model loaded successfully")


def main():
    parser = argparse.ArgumentParser(description='Melanoma GAN for synthetic data generation using PyTorch')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing malignant melanoma images')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], required=True,
                        help='Mode: train the model or generate synthetic images')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train (for train mode)')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Size to resize images (square)')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of synthetic images to generate (for generate mode)')
    parser.add_argument('--output_dir', type=str, default='generated_melanoma',
                        help='Directory to save generated images')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to pre-trained generator model to load (for generate mode)')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Number of steps for gradient accumulation')

    args = parser.parse_args()

    # Initialize the GAN
    gan = MelanomaGAN(args.data_dir, batch_size=args.batch_size, img_size=args.img_size)

    if args.mode == 'train':
        # Train the model
        gan.train(epochs=args.epochs, accumulation_steps=args.accumulation_steps)
    elif args.mode == 'generate':
        if args.load_model:
            # Load pre-trained model
            gan.load_model(args.load_model)

        # Generate synthetic images
        gan.generate_images(num_images=args.num_images, output_dir=args.output_dir)


if __name__ == '__main__':
    main()