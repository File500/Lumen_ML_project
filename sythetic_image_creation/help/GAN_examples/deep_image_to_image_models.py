#!/usr/bin/env python

# pylint: disable=arguments-differ,unused-import

import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
                                     Flatten, SpatialDropout2D)
from tensorflow.nn import leaky_relu, tanh

from deep_model_blocks import (BottleneckResidualBlock, Conv, ConvBlock,
                               Deconv, DeconvBlock, PreActivationResidualBlock,
                               ResidualBlock, ReverseBottleneckResidualBlock,
                               ReverseResidualBlock, UBlock)
from model import Model


class ResidualThreeStridesTwoBlocks(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(ResidualThreeStridesTwoBlocks.Generator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ResidualBlock(initial_filters*1, 7, 2),
          ResidualBlock(initial_filters*1, 3, 1),

          ResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*2, 3, 1),

          ResidualBlock(initial_filters*4, 3, 2),
          ResidualBlock(initial_filters*4, 3, 1),

          ResidualBlock(initial_filters*8, 3, 1),
          ResidualBlock(initial_filters*8, 3, 1),

          ReverseResidualBlock(initial_filters*4, 3, 1),
          ReverseResidualBlock(initial_filters*4, 3, 2),

          ReverseResidualBlock(initial_filters*2, 3, 1),
          ReverseResidualBlock(initial_filters*2, 3, 2),

          ReverseResidualBlock(initial_filters*1, 3, 1),
          ReverseResidualBlock(initial_filters*1, 3, 2)
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(ResidualThreeStridesTwoBlocks.Discriminator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ResidualBlock(initial_filters*1, 7, 2),
          ResidualBlock(initial_filters*1, 3, 1),

          ResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*2, 3, 1),

          ResidualBlock(initial_filters*4, 3, 2),
          ResidualBlock(initial_filters*4, 3, 1),

          ResidualBlock(initial_filters*8, 3, 2),
          ResidualBlock(initial_filters*8, 3, 1),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class ResidualThreeStridesThreeBlocks(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(ResidualThreeStridesThreeBlocks.Generator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ResidualBlock(initial_filters*1, 7, 2),
          ResidualBlock(initial_filters*1, 3, 1),
          ResidualBlock(initial_filters*1, 3, 1),

          ResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*2, 3, 1),
          ResidualBlock(initial_filters*2, 3, 1),

          ResidualBlock(initial_filters*4, 3, 2),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),

          ResidualBlock(initial_filters*8, 3, 1),
          ResidualBlock(initial_filters*8, 3, 1),
          ResidualBlock(initial_filters*8, 3, 1),

          ReverseResidualBlock(initial_filters*4, 3, 1),
          ReverseResidualBlock(initial_filters*4, 3, 1),
          ReverseResidualBlock(initial_filters*4, 3, 2),

          ReverseResidualBlock(initial_filters*2, 3, 1),
          ReverseResidualBlock(initial_filters*2, 3, 1),
          ReverseResidualBlock(initial_filters*2, 3, 2),

          ReverseResidualBlock(initial_filters*1, 3, 1),
          ReverseResidualBlock(initial_filters*1, 3, 1),
          ReverseResidualBlock(initial_filters*1, 3, 2)
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(ResidualThreeStridesThreeBlocks.Discriminator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ResidualBlock(initial_filters*1, 7, 2),
          ResidualBlock(initial_filters*1, 3, 1),
          ResidualBlock(initial_filters*1, 3, 1),

          ResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*2, 3, 1),
          ResidualBlock(initial_filters*2, 3, 1),

          ResidualBlock(initial_filters*4, 3, 2),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),

          ResidualBlock(initial_filters*8, 3, 2),
          ResidualBlock(initial_filters*8, 3, 1),
          ResidualBlock(initial_filters*8, 3, 1),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class ResidualThreeStridesMoreFilters(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(ResidualThreeStridesMoreFilters.Generator, self).__init__()

      # this is pretty similar to CycleGAN which has conv with 32 7x7 filters, then 2 strided convs
      # with 64 and 128 3x3 filters, then 9 residual blocks with 128 filters, then 2 strided deconvs
      # with 64 and 32 3x3 filters, then finally a convolution with 3 7x7 filters

      initial_filters = 64

      self.initial_conv = Conv(initial_filters*1, 7, 1)
      self.initial_batchnorm = BatchNormalization()
      self.blocks = [
          ResidualBlock(initial_filters*1, 3, 2),
          ResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*4, 3, 2),

          ResidualBlock(initial_filters*8, 3, 1),
          ResidualBlock(initial_filters*8, 3, 1),
          ResidualBlock(initial_filters*8, 3, 1),
          ResidualBlock(initial_filters*8, 3, 1),

          ReverseResidualBlock(initial_filters*4, 3, 2),
          ReverseResidualBlock(initial_filters*2, 3, 2),
          ReverseResidualBlock(initial_filters*1, 3, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      x = self.initial_conv(x)
      x = self.initial_batchnorm(x, training=training)
      x = leaky_relu(x)
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(ResidualThreeStridesMoreFilters.Discriminator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ResidualBlock(initial_filters*1, 7, 2),
          ResidualBlock(initial_filters*1, 3, 1),

          ResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*2, 3, 1),

          ResidualBlock(initial_filters*4, 3, 2),
          ResidualBlock(initial_filters*4, 3, 1),

          ResidualBlock(initial_filters*8, 3, 2),
          ResidualBlock(initial_filters*8, 3, 1),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class ResidualThreeStridesDeeper(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(ResidualThreeStridesDeeper.Generator, self).__init__()

      initial_filters = 64

      self.initial_conv = Conv(initial_filters*1, 7, 1)
      self.initial_batchnorm = BatchNormalization()
      self.blocks = [
          ResidualBlock(initial_filters*1, 3, 2),
          ResidualBlock(initial_filters*1, 3, 1),
          ResidualBlock(initial_filters*1, 3, 1),
          ResidualBlock(initial_filters*1, 3, 1),

          ResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*2, 3, 1),
          ResidualBlock(initial_filters*2, 3, 1),
          ResidualBlock(initial_filters*2, 3, 1),

          ResidualBlock(initial_filters*4, 3, 2),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),

          ReverseResidualBlock(initial_filters*4, 3, 1),
          ReverseResidualBlock(initial_filters*4, 3, 1),
          ReverseResidualBlock(initial_filters*4, 3, 1),
          ReverseResidualBlock(initial_filters*4, 3, 2),

          ReverseResidualBlock(initial_filters*2, 3, 1),
          ReverseResidualBlock(initial_filters*2, 3, 1),
          ReverseResidualBlock(initial_filters*2, 3, 1),
          ReverseResidualBlock(initial_filters*2, 3, 2),

          ReverseResidualBlock(initial_filters*1, 3, 1),
          ReverseResidualBlock(initial_filters*1, 3, 1),
          ReverseResidualBlock(initial_filters*1, 3, 1),
          ReverseResidualBlock(initial_filters*1, 3, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      x = self.initial_conv(x)
      x = self.initial_batchnorm(x, training=training)
      x = leaky_relu(x)
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(ResidualThreeStridesDeeper.Discriminator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ResidualBlock(initial_filters*1, 7, 2),
          ResidualBlock(initial_filters*1, 3, 1),
          ResidualBlock(initial_filters*1, 3, 1),
          ResidualBlock(initial_filters*1, 3, 1),

          ResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*2, 3, 1),
          ResidualBlock(initial_filters*2, 3, 1),
          ResidualBlock(initial_filters*2, 3, 1),

          ResidualBlock(initial_filters*4, 3, 2),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class CycleGan(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(CycleGan.Generator, self).__init__()

      initial_filters = 32

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 1),
          ConvBlock(initial_filters*2, 3, 2),
          ConvBlock(initial_filters*4, 3, 2),

          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),

          DeconvBlock(initial_filters*2, 3, 2),
          DeconvBlock(initial_filters*1, 3, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(CycleGan.Discriminator, self).__init__()

      # this should actually be a 70x70 PatchGAN

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 4, 2),
          ConvBlock(initial_filters*2, 4, 2),
          ConvBlock(initial_filters*4, 4, 2),
          ConvBlock(initial_filters*8, 4, 2),
          ]

      self.final_conv = Conv(1, 4, 1)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        # no dropout!
      x = self.final_conv(x)
      x = self.flatten(x)
      return self.fc(x)


class ResidualOneStride(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(ResidualOneStride.Generator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 1),
          ResidualBlock(initial_filters*1, 3, 2),
          ResidualBlock(initial_filters*2, 3, 1),
          ResidualBlock(initial_filters*2, 3, 1),
          ResidualBlock(initial_filters*4, 3, 1),
          ReverseResidualBlock(initial_filters*4, 3, 1),
          ReverseResidualBlock(initial_filters*2, 3, 1),
          ReverseResidualBlock(initial_filters*2, 3, 1),
          ReverseResidualBlock(initial_filters*1, 3, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(ResidualOneStride.Discriminator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 2),
          ResidualBlock(initial_filters*1, 3, 1),
          ResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*2, 3, 1),
          ResidualBlock(initial_filters*4, 3, 2),
          ResidualBlock(initial_filters*4, 3, 1),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class ResidualAlternatingStrides(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(ResidualAlternatingStrides.Generator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 1),
          ConvBlock(initial_filters*1, 3, 2),

          ResidualBlock(initial_filters*2, 3, 2),
          ReverseResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*4, 3, 2),
          ReverseResidualBlock(initial_filters*8, 3, 2),
          ResidualBlock(initial_filters*8, 3, 2),
          ReverseResidualBlock(initial_filters*4, 3, 2),
          ResidualBlock(initial_filters*2, 3, 2),
          ReverseResidualBlock(initial_filters*2, 3, 2),

          DeconvBlock(initial_filters*1, 3, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(ResidualAlternatingStrides.Discriminator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 2),
          ResidualBlock(initial_filters*1, 3, 1),
          ResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*2, 3, 1),
          ResidualBlock(initial_filters*4, 3, 2),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*8, 3, 2),
          ResidualBlock(initial_filters*8, 3, 1),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class BottleneckResidualAlternatingStrides(Model):
  # NOTE: this is exactly the same as "ResidualAlternatingStrides" even though more capacity would be available
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(BottleneckResidualAlternatingStrides.Generator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 1),
          ConvBlock(initial_filters*1, 3, 2),

          BottleneckResidualBlock(initial_filters*2, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*2, 3, 2),
          BottleneckResidualBlock(initial_filters*4, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*8, 3, 2),
          BottleneckResidualBlock(initial_filters*8, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*4, 3, 2),
          BottleneckResidualBlock(initial_filters*2, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*2, 3, 2),

          DeconvBlock(initial_filters*1, 3, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(BottleneckResidualAlternatingStrides.Discriminator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 2),
          BottleneckResidualBlock(initial_filters*1, 3, 1),
          BottleneckResidualBlock(initial_filters*2, 3, 2),
          BottleneckResidualBlock(initial_filters*2, 3, 1),
          BottleneckResidualBlock(initial_filters*4, 3, 2),
          BottleneckResidualBlock(initial_filters*4, 3, 1),
          BottleneckResidualBlock(initial_filters*8, 3, 2),
          BottleneckResidualBlock(initial_filters*8, 3, 1),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class BottleneckResidualAlternatingTwoStrides(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(BottleneckResidualAlternatingTwoStrides.Generator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 1),
          ConvBlock(initial_filters*1, 3, 2),

          BottleneckResidualBlock(initial_filters*8, 3, 2),
          BottleneckResidualBlock(initial_filters*16, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*16, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*8, 3, 2),
          BottleneckResidualBlock(initial_filters*8, 3, 2),
          BottleneckResidualBlock(initial_filters*16, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*16, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*8, 3, 2),
          BottleneckResidualBlock(initial_filters*8, 3, 2),
          BottleneckResidualBlock(initial_filters*16, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*16, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*8, 3, 2),

          DeconvBlock(initial_filters*1, 3, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(BottleneckResidualAlternatingTwoStrides.Discriminator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 2),
          BottleneckResidualBlock(initial_filters*1, 3, 1),
          BottleneckResidualBlock(initial_filters*2, 3, 2),
          BottleneckResidualBlock(initial_filters*2, 3, 1),
          BottleneckResidualBlock(initial_filters*4, 3, 2),
          BottleneckResidualBlock(initial_filters*4, 3, 1),
          BottleneckResidualBlock(initial_filters*8, 3, 2),
          BottleneckResidualBlock(initial_filters*8, 3, 1),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class LargerBottleneckResidualAlternatingStrides(Model):
  # NOTE: this is exactly the same as "ResidualAlternatingStrides" even though more capacity would be available
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(LargerBottleneckResidualAlternatingStrides.Generator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 1),
          ConvBlock(initial_filters*1, 3, 2),

          BottleneckResidualBlock(initial_filters*2, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*2, 3, 2),
          BottleneckResidualBlock(initial_filters*4, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*4, 3, 2),
          BottleneckResidualBlock(initial_filters*8, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*8, 3, 2),
          BottleneckResidualBlock(initial_filters*4, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*4, 3, 2),
          BottleneckResidualBlock(initial_filters*2, 3, 2),
          ReverseBottleneckResidualBlock(initial_filters*2, 3, 2),

          DeconvBlock(initial_filters*1, 3, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(LargerBottleneckResidualAlternatingStrides.Discriminator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 2),
          BottleneckResidualBlock(initial_filters*1, 3, 1),
          BottleneckResidualBlock(initial_filters*2, 3, 2),
          BottleneckResidualBlock(initial_filters*2, 3, 1),
          BottleneckResidualBlock(initial_filters*4, 3, 2),
          BottleneckResidualBlock(initial_filters*4, 3, 1),
          BottleneckResidualBlock(initial_filters*8, 3, 2),
          BottleneckResidualBlock(initial_filters*8, 3, 1),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class ShallowUBlocks(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(ShallowUBlocks.Generator, self).__init__()

      self.blocks = [
          ConvBlock(64),
          UBlock(64, 512, 4),
          UBlock(64, 512, 4),
          UBlock(64, 512, 4),
          UBlock(64, 512, 4),
          UBlock(64, 512, 4),
          UBlock(64, 512, 4),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(ShallowUBlocks.Discriminator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 2),
          ResidualBlock(initial_filters*1, 3, 1),
          ResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*2, 3, 1),
          ResidualBlock(initial_filters*4, 3, 2),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*8, 3, 2),
          ResidualBlock(initial_filters*8, 3, 1),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class DeepUBlocks(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(DeepUBlocks.Generator, self).__init__()

      self.blocks = [
          ConvBlock(128),
          UBlock(128, 512, 6),
          UBlock(128, 512, 6),
          UBlock(128, 512, 6),
          UBlock(128, 512, 6),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 7, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(DeepUBlocks.Discriminator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 7, 2),
          ResidualBlock(initial_filters*1, 3, 1),
          ResidualBlock(initial_filters*2, 3, 2),
          ResidualBlock(initial_filters*2, 3, 1),
          ResidualBlock(initial_filters*4, 3, 2),
          ResidualBlock(initial_filters*4, 3, 1),
          ResidualBlock(initial_filters*8, 3, 2),
          ResidualBlock(initial_filters*8, 3, 1),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      x = self.fc(x)
      return x


class SimpleResidual(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(SimpleResidual.Generator, self).__init__()

      initial_filters = 32*2

      self.blocks = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*4, 5, 2),

          ResidualBlock(initial_filters*8, 5, 1),
          ResidualBlock(initial_filters*8, 5, 1),
          ResidualBlock(initial_filters*8, 5, 1),
          ResidualBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          DeconvBlock(initial_filters*2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(SimpleResidual.Discriminator, self).__init__()

      initial_filters = 64*2

      self.blocks = [
          ConvBlock(initial_filters*1, 4, 2),
          ConvBlock(initial_filters*2, 4, 2),
          ConvBlock(initial_filters*4, 4, 2),
          ConvBlock(initial_filters*8, 4, 2),
          ConvBlock(initial_filters*16, 4, 2),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes, use_bias=False)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      return self.fc(x)


class SkipResidual(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(SkipResidual.Generator, self).__init__()

      initial_filters = 32*2

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*4, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*4, 5, 2),
          DeconvBlock(initial_filters*2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      e0 = x
      e1 = self.encoder[0](x, training=training)
      e2 = self.encoder[1](e1, training=training)
      x = self.encoder[2](e2, training=training)

      for block in self.res:
        x = block(x, training=training)

      x = self.decoder[0](x, training=training)
      x = tf.concat([x, e2], axis=-1)
      x = self.decoder[1](x, training=training)
      x = tf.concat([x, e1], axis=-1)
      x = self.decoder[2](x, training=training)
      x = tf.concat([x, e0], axis=-1)

      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(SkipResidual.Discriminator, self).__init__()

      initial_filters = 64*2*2

      self.blocks = [
          ConvBlock(initial_filters*1, 4, 2),
          ConvBlock(initial_filters*2, 4, 2),
          ConvBlock(initial_filters*4, 4, 2),
          ConvBlock(initial_filters*8, 4, 2),
          # ConvBlock(initial_filters*16, 4, 2),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes, use_bias=False)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      return self.fc(x)


class SkipResidualDropout(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(SkipResidualDropout.Generator, self).__init__()

      initial_filters = 32*2

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*4, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*4, 5, 2),
          DeconvBlock(initial_filters*2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)
      # self.dropout = Dropout(0.1)

    def call(self, x, training=True):
      e0 = x
      e1 = self.encoder[0](x, training=training)
      e2 = self.encoder[1](e1, training=training)
      x = self.encoder[2](e2, training=training)

      # x = self.dropout(x, training=training)

      for block in self.res:
        x = block(x, training=training)

      x = self.decoder[0](x, training=training)
      x = tf.concat([x, e2], axis=-1)
      x = self.decoder[1](x, training=training)
      x = tf.concat([x, e1], axis=-1)
      x = self.decoder[2](x, training=training)
      x = tf.concat([x, e0], axis=-1)

      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(SkipResidualDropout.Discriminator, self).__init__()

      initial_filters = 64*2*2

      self.blocks = [
          ConvBlock(initial_filters*1, 4, 2),
          ConvBlock(initial_filters*2, 4, 2),
          ConvBlock(initial_filters*4, 4, 2),
          ConvBlock(initial_filters*8, 4, 2),
          # ConvBlock(initial_filters*16, 4, 2),
          ]

      self.dropout = Dropout(0.5)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes, use_bias=False)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      return self.fc(x)


class NoSkipTwoStrideMsDisc(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(NoSkipTwoStrideMsDisc.Generator, self).__init__()

      tf.logging.fatal("Not using any skip connections!")

      initial_filters = 32*1

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      x = self.encoder[0](x, training=training)
      x = self.encoder[1](x, training=training)

      for block in self.res:
        x = block(x, training=training)

      x = self.decoder[0](x, training=training)
      x = self.decoder[1](x, training=training)

      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout, resolution):
        super(NoSkipTwoStrideMsDisc.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(resolution * scaling_factor)
          size_y = int(resolution * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            # ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(NoSkipTwoStrideMsDisc.Discriminator, self).__init__()
      resolution = 256//2
      tf.logging.fatal("Using MS disc for {0}x{0} patches!".format(resolution))
      self.discriminators = [NoSkipTwoStrideMsDisc.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3), resolution) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(NoSkipTwoStrideMsDisc.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class ConcatSkipThreeStrideMsDisc(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(ConcatSkipThreeStrideMsDisc.Generator, self).__init__()

      tf.logging.fatal("Concatenating skip connections!")

      initial_filters = 32*1

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*4, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*4, 5, 2),
          DeconvBlock(initial_filters*2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      e0 = x
      e1 = self.encoder[0](x, training=training)
      e2 = self.encoder[1](e1, training=training)
      x = self.encoder[2](e2, training=training)

      for block in self.res:
        x = block(x, training=training)

      x = self.decoder[0](x, training=training)
      x = tf.concat([x, e2], axis=-1)
      x = self.decoder[1](x, training=training)
      x = tf.concat([x, e1], axis=-1)
      x = self.decoder[2](x, training=training)
      x = tf.concat([x, e0], axis=-1)

      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout, resolution):
        super(ConcatSkipThreeStrideMsDisc.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(resolution * scaling_factor)
          size_y = int(resolution * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            # ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(ConcatSkipThreeStrideMsDisc.Discriminator, self).__init__()
      resolution = 256//2
      tf.logging.fatal("Using MS disc for {0}x{0} patches!".format(resolution))
      self.discriminators = [ConcatSkipThreeStrideMsDisc.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3), resolution) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(ConcatSkipThreeStrideMsDisc.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)

class ConcatSkipTwoStrideMsDisc(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(ConcatSkipTwoStrideMsDisc.Generator, self).__init__()

      tf.logging.fatal("Concatenating skip connections!")

      initial_filters = 32*1

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      e0 = x
      e1 = self.encoder[0](x, training=training)
      e2 = self.encoder[1](e1, training=training)
      x = e2

      for block in self.res:
        x = block(x, training=training)

      x = tf.concat([x, e2], axis=-1)
      x = self.decoder[0](x, training=training)
      x = tf.concat([x, e1], axis=-1)
      x = self.decoder[1](x, training=training)
      x = tf.concat([x, e0], axis=-1)

      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout, resolution):
        super(ConcatSkipTwoStrideMsDisc.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(resolution * scaling_factor)
          size_y = int(resolution * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            # ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(ConcatSkipTwoStrideMsDisc.Discriminator, self).__init__()
      resolution = 256//2
      tf.logging.fatal("Using MS disc for {0}x{0} patches!".format(resolution))
      self.discriminators = [ConcatSkipTwoStrideMsDisc.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3), resolution) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(ConcatSkipTwoStrideMsDisc.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)

class ConcatAddSkipTwoStrideMsDisc(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(ConcatAddSkipTwoStrideMsDisc.Generator, self).__init__()

      tf.logging.fatal("Concatenating skip connections and adding to output!")

      initial_filters = 32*1

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      images = x[:, :, :, :3]
      e0 = x
      e1 = self.encoder[0](x, training=training)
      e2 = self.encoder[1](e1, training=training)
      x = e2

      for block in self.res:
        x = block(x, training=training)

      x = tf.concat([x, e2], axis=-1)
      x = self.decoder[0](x, training=training)
      x = tf.concat([x, e1], axis=-1)
      x = self.decoder[1](x, training=training)
      x = tf.concat([x, e0], axis=-1)

      return tf.minimum(tf.maximum(tf.add(tanh(self.final_conv(x)), images), -1), 1)

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout, resolution):
        super(ConcatAddSkipTwoStrideMsDisc.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(resolution * scaling_factor)
          size_y = int(resolution * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            # ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(ConcatAddSkipTwoStrideMsDisc.Discriminator, self).__init__()
      resolution = 256//2
      tf.logging.fatal("Using MS disc for {0}x{0} patches!".format(resolution))
      self.discriminators = [ConcatAddSkipTwoStrideMsDisc.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3), resolution) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(ConcatAddSkipTwoStrideMsDisc.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class AddSkipThreeStrideMsDisc(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(AddSkipThreeStrideMsDisc.Generator, self).__init__()

      tf.logging.fatal("Adding skip connections!")

      initial_filters = 32*1

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*4, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8//2, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*4//2, 5, 2),
          DeconvBlock(initial_filters*2//2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      e0 = self.encoder[0](x, training=training)
      e1 = self.encoder[1](e0, training=training)
      e2 = self.encoder[2](e1, training=training)
      x = e2

      for block in self.res:
        x = block(x, training=training)

      x = tf.add(x, e2)
      x = self.decoder[0](x, training=training)
      x = tf.add(x, e1)
      x = self.decoder[1](x, training=training)
      x = tf.add(x, e0)
      x = self.decoder[2](x, training=training)

      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout, resolution):
        super(AddSkipThreeStrideMsDisc.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(resolution * scaling_factor)
          size_y = int(resolution * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            # ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(AddSkipThreeStrideMsDisc.Discriminator, self).__init__()
      resolution = 256//2
      tf.logging.fatal("Using MS disc for {0}x{0} patches!".format(resolution))
      self.discriminators = [AddSkipThreeStrideMsDisc.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3), resolution) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(AddSkipThreeStrideMsDisc.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class AddSkipTwoStrideMsDisc(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(AddSkipTwoStrideMsDisc.Generator, self).__init__()

      tf.logging.fatal("Adding skip connections!")

      initial_filters = 32*1

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4//2, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*2//2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      e0 = self.encoder[0](x, training=training)
      e1 = self.encoder[1](e0, training=training)
      x = e1

      for block in self.res:
        x = block(x, training=training)

      x = tf.add(x, e1)
      x = self.decoder[0](x, training=training)
      x = tf.add(x, e0)
      x = self.decoder[1](x, training=training)

      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout, resolution):
        super(AddSkipTwoStrideMsDisc.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(resolution * scaling_factor)
          size_y = int(resolution * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            # ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(AddSkipTwoStrideMsDisc.Discriminator, self).__init__()
      resolution = 256//2
      tf.logging.fatal("Using MS disc for {0}x{0} patches!".format(resolution))
      self.discriminators = [AddSkipTwoStrideMsDisc.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3), resolution) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(AddSkipTwoStrideMsDisc.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class AddSkipToOutputThreeStrideMsDisc(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(AddSkipToOutputThreeStrideMsDisc.Generator, self).__init__()

      tf.logging.fatal("Adding skip connections, including output!")

      initial_filters = 32*1

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*4, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8//2, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*4//2, 5, 2),
          DeconvBlock(initial_filters*2//2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      images = x[:, :, :, :3]
      e0 = self.encoder[0](x, training=training)
      e1 = self.encoder[1](e0, training=training)
      e2 = self.encoder[2](e1, training=training)
      x = e2

      for block in self.res:
        x = block(x, training=training)

      x = tf.add(x, e2)
      x = self.decoder[0](x, training=training)
      x = tf.add(x, e1)
      x = self.decoder[1](x, training=training)
      x = tf.add(x, e0)
      x = self.decoder[2](x, training=training)

      return tf.minimum(tf.maximum(tf.add(tanh(self.final_conv(x)), images), -1), 1)

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout, resolution):
        super(AddSkipToOutputThreeStrideMsDisc.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(resolution * scaling_factor)
          size_y = int(resolution * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            # ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(AddSkipToOutputThreeStrideMsDisc.Discriminator, self).__init__()
      resolution = 256//2
      tf.logging.fatal("Using MS disc for {0}x{0} patches!".format(resolution))
      self.discriminators = [AddSkipToOutputThreeStrideMsDisc.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3), resolution) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(AddSkipToOutputThreeStrideMsDisc.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class AddSkipToOutputTwoStrideMsDisc(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(AddSkipToOutputTwoStrideMsDisc.Generator, self).__init__()

      tf.logging.fatal("Adding skip connections, including output!")

      initial_filters = 32*1

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4//2, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*2//2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      images = x[:, :, :, :3]
      e0 = self.encoder[0](x, training=training)
      e1 = self.encoder[1](e0, training=training)
      x = e1

      for block in self.res:
        x = block(x, training=training)

      x = tf.add(x, e1)
      x = self.decoder[0](x, training=training)
      x = tf.add(x, e0)
      x = self.decoder[1](x, training=training)

      return tf.minimum(tf.maximum(tf.add(tanh(self.final_conv(x)), images), -1), 1)

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout, resolution):
        super(AddSkipToOutputTwoStrideMsDisc.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(resolution * scaling_factor)
          size_y = int(resolution * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            # ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(AddSkipToOutputTwoStrideMsDisc.Discriminator, self).__init__()
      resolution = 256//2
      tf.logging.fatal("Using MS disc for {0}x{0} patches!".format(resolution))
      self.discriminators = [AddSkipToOutputTwoStrideMsDisc.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3), resolution) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(AddSkipToOutputTwoStrideMsDisc.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class AddSkipToOutputTwoStrideMsDiscAlternative(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(AddSkipToOutputTwoStrideMsDiscAlternative.Generator, self).__init__()

      tf.logging.fatal("Adding skip connections, including output!")

      initial_filters = 32*1

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4//2, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*2//2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      images = x[:, :, :, :3]
      e0 = self.encoder[0](x, training=training)
      e1 = self.encoder[1](e0, training=training)
      x = e1

      for block in self.res:
        x = block(x, training=training)

      x = tf.add(x, e1)
      x = self.decoder[0](x, training=training)
      x = tf.add(x, e0)
      x = self.decoder[1](x, training=training)

      return tf.minimum(tf.maximum(tf.add(tanh(self.final_conv(x)), images), -1), 1)

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout, resolution):
        super(AddSkipToOutputTwoStrideMsDiscAlternative.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(resolution * scaling_factor)
          size_y = int(resolution * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            # ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(AddSkipToOutputTwoStrideMsDiscAlternative.Discriminator, self).__init__()
      resolution = 256//2
      tf.logging.fatal("Using MS disc for {0}x{0} patches!".format(resolution))
      self.discriminators = [AddSkipToOutputTwoStrideMsDiscAlternative.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3), resolution) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(AddSkipToOutputTwoStrideMsDiscAlternative.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class SkipResidualMsDiscPathoSeparate(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(SkipResidualMsDiscPathoSeparate.Generator, self).__init__()

      initial_filters = 32*1

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*4, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      patho = x
      patho_2 = tf.image.resize_nearest_neighbor(patho, (patho.shape[1]//2, patho.shape[2]//2))
      patho_4 = tf.image.resize_nearest_neighbor(patho, (patho.shape[1]//4, patho.shape[2]//4))

      e0 = x
      e1 = self.encoder[0](tf.concat([x, patho], axis=-1), training=training)
      x = self.encoder[1](tf.concat([e1, patho_2], axis=-1), training=training)

      for block in self.res:
        x = block(tf.concat([x, patho_4], axis=-1), training=training)

      x = self.decoder[0](x, training=training)
      x = tf.concat([x, e1, patho_2], axis=-1)
      x = self.decoder[1](x, training=training)
      x = tf.concat([x, e0, patho], axis=-1)

      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout, resolution):
        super(SkipResidualMsDiscPathoSeparate.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(resolution * scaling_factor)
          size_y = int(resolution * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*1

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            # ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(SkipResidualMsDiscPathoSeparate.Discriminator, self).__init__()
      resolution = 256//2
      tf.logging.fatal("Using MS disc for {0}x{0} patches!".format(resolution))
      self.discriminators = [SkipResidualMsDiscPathoSeparate.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3), resolution) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(SkipResidualMsDiscPathoSeparate.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class SkipResidualMsDiscGLowRes(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(SkipResidualMsDiscGLowRes.Generator, self).__init__()

      initial_filters = 32*2

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*4, 5, 2),
          ConvBlock(initial_filters*8, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*8, 5, 2),
          DeconvBlock(initial_filters*4, 5, 2),
          DeconvBlock(initial_filters*2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      e0 = x
      e1 = self.encoder[0](x, training=training)
      e2 = self.encoder[1](e1, training=training)
      e3 = self.encoder[2](e2, training=training)
      x = self.encoder[3](e3, training=training)

      for block in self.res:
        x = block(x, training=training)

      x = self.decoder[0](x, training=training)
      x = tf.concat([x, e3], axis=-1)
      x = self.decoder[1](x, training=training)
      x = tf.concat([x, e2], axis=-1)
      x = self.decoder[2](x, training=training)
      x = tf.concat([x, e1], axis=-1)
      x = self.decoder[3](x, training=training)
      x = tf.concat([x, e0], axis=-1)

      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout, resolution):
        super(SkipResidualMsDiscGLowRes.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(resolution * scaling_factor)
          size_y = int(resolution * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*2

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(SkipResidualMsDiscGLowRes.Discriminator, self).__init__()
      resolution = 256
      tf.logging.fatal("Using MS disc for {0}x{0} patches!".format(resolution))
      self.discriminators = [SkipResidualMsDiscGLowRes.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3), resolution) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(SkipResidualMsDiscGLowRes.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class SkipResidualMsDiscGHighRes(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(SkipResidualMsDiscGHighRes.Generator, self).__init__()

      initial_filters = 32*2

      self.encoder = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ResidualBlock(initial_filters*8, 5, 1, project_shortcut=True),
          ]

      self.decoder = [
          DeconvBlock(initial_filters*2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      e0 = x
      e1 = self.encoder[0](x, training=training)
      x = self.encoder[1](e1, training=training)

      for block in self.res:
        x = block(x, training=training)

      x = self.decoder[0](x, training=training)
      x = tf.concat([x, e1], axis=-1)
      x = self.decoder[1](x, training=training)
      x = tf.concat([x, e0], axis=-1)

      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    class MultiscaleDisc(tf.keras.Model):
      def __init__(self, config, scaling_factor, dropout, resolution):
        super(SkipResidualMsDiscGHighRes.Discriminator.MultiscaleDisc, self).__init__()

        assert scaling_factor > 0
        if scaling_factor != 1:
          size_x = int(resolution * scaling_factor)
          size_y = int(resolution * scaling_factor)
          tf.logging.info("Multiscale discriminator operating on resolution: {}x{}".format(size_x, size_y))
          self.resize = lambda x: tf.image.resize_nearest_neighbor(x, (size_x, size_y))
        else:
          tf.logging.info("Multiscale discriminator operating on regular resolution")
          self.resize = lambda x: x

        initial_filters = 32*2

        self.blocks = [
            ConvBlock(initial_filters*1, 4, 2),
            ConvBlock(initial_filters*2, 4, 2),
            ConvBlock(initial_filters*4, 4, 2),
            ConvBlock(initial_filters*8, 4, 2),
            ConvBlock(initial_filters*16, 4, 2),
            ]

        self.dropout = dropout
        self.flatten = Flatten()
        self.fc = Dense(config.discriminator_classes, use_bias=False)

      def call(self, x, training):
        x = self.resize(x)
        for block in self.blocks:
          x = block(x, training=training)
          x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def __init__(self, config):
      super(SkipResidualMsDiscGHighRes.Discriminator, self).__init__()
      resolution = 256
      tf.logging.fatal("Using MS disc for {0}x{0} patches!".format(resolution))
      self.discriminators = [SkipResidualMsDiscGHighRes.Discriminator.MultiscaleDisc(
        config, factor, Dropout(0.3), resolution) for factor in [1, 0.5]]

    def call(self, x, training=True):
      return tf.reduce_mean(tf.concat([disc(x, training) for disc in self.discriminators], axis=-1), axis=-1)

    def summary(self, line_length=None, positions=None, print_fn=None):
      super(SkipResidualMsDiscGHighRes.Discriminator, self).summary(line_length, positions, print_fn)
      print_fn("\nDetails:")
      for discriminator in self.discriminators:
        discriminator.summary(line_length, positions, print_fn)


class UNet(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(UNet.Generator, self).__init__()

      initial_filters = 32*2

      self.encoder_unstrided = [
          ConvBlock(initial_filters*1, 5, 1),
          ConvBlock(initial_filters*2, 5, 1),
          ConvBlock(initial_filters*4, 5, 1),
          ]
      self.encoder_strided = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*4, 5, 2),
          ]

      self.res = [
          ResidualBlock(initial_filters*8, 5, 1),
          ResidualBlock(initial_filters*8, 5, 1),
          ResidualBlock(initial_filters*8, 5, 1),
          ResidualBlock(initial_filters*8, 5, 1),
          ]

      self.decoder_strided = [
          DeconvBlock(initial_filters*4, 5, 2),
          DeconvBlock(initial_filters*2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.decoder_unstrided = [
          DeconvBlock(initial_filters*4, 5, 1),
          DeconvBlock(initial_filters*2, 5, 1),
          DeconvBlock(initial_filters*1, 5, 1),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      e0 = self.encoder_unstrided[0](x, training=training)

      e1 = self.encoder_strided[0](e0, training=training)
      e1 = self.encoder_unstrided[1](e1, training=training)

      e2 = self.encoder_strided[1](e1, training=training)
      e2 = self.encoder_unstrided[2](e2, training=training)

      x = self.encoder_strided[2](e2, training=training)

      for block in self.res:
        x = block(x, training=training)

      x = self.decoder_strided[0](x, training=training)

      x = tf.concat([x, e2], axis=-1)
      x = self.decoder_unstrided[0](x, training=training)

      x = self.decoder_strided[1](x, training=training)
      x = tf.concat([x, e1], axis=-1)
      x = self.decoder_unstrided[1](x, training=training)

      x = self.decoder_strided[2](x, training=training)
      x = tf.concat([x, e0], axis=-1)
      x = self.decoder_unstrided[2](x, training=training)

      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(UNet.Discriminator, self).__init__()

      initial_filters = 64*2

      self.blocks = [
          ConvBlock(initial_filters*1, 4, 2),
          ConvBlock(initial_filters*2, 4, 2),
          ConvBlock(initial_filters*4, 4, 2),
          ConvBlock(initial_filters*8, 4, 2),
          ConvBlock(initial_filters*16, 4, 2),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes, use_bias=False)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      return self.fc(x)


class SimpleConv(Model):
  class Generator(tf.keras.Model):
    def __init__(self, config):
      super(SimpleConv.Generator, self).__init__()

      initial_filters = 32*2

      self.blocks = [
          ConvBlock(initial_filters*1, 5, 2),
          ConvBlock(initial_filters*2, 5, 2),
          ConvBlock(initial_filters*4, 5, 2),

          ConvBlock(initial_filters*8, 5, 1),
          ConvBlock(initial_filters*8, 5, 1),
          ConvBlock(initial_filters*8, 5, 1),
          ConvBlock(initial_filters*8, 5, 1),

          DeconvBlock(initial_filters*4, 5, 2),
          DeconvBlock(initial_filters*2, 5, 2),
          DeconvBlock(initial_filters*1, 5, 2),
          ]
      self.final_conv = Conv(3 if config.has_colored_target else 1, 5, 1)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
      return tanh(self.final_conv(x))

  class Discriminator(tf.keras.Model):
    def __init__(self, config):
      super(SimpleConv.Discriminator, self).__init__()

      initial_filters = 64

      self.blocks = [
          ConvBlock(initial_filters*1, 4, 2),
          ConvBlock(initial_filters*2, 4, 2),
          ConvBlock(initial_filters*4, 4, 2),
          ConvBlock(initial_filters*8, 4, 2),
          ConvBlock(initial_filters*16, 4, 2),
          ]

      self.dropout = Dropout(0.3)
      self.flatten = Flatten()
      self.fc = Dense(config.discriminator_classes, use_bias=False)

    def call(self, x, training=True):
      for block in self.blocks:
        x = block(x, training=training)
        x = self.dropout(x, training=training)
      x = self.flatten(x)
      return self.fc(x)
