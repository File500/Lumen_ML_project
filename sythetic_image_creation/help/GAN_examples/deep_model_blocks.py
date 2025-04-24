#!/usr/bin/env python

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.eager import context
from tensorflow.python.layers import utils
from tensorflow.python.layers import convolutional as tf_convolutional_layers
from tensorflow.python.util.tf_export import tf_export
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer
from tensorflow.python.layers import core
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import gen_math_ops

def _l2normalizer(v, epsilon=1e-12):
    return v / (K.sum(v ** 2) ** 0.5 + epsilon)


def power_iteration(W, u, rounds=1):
    """
    Accroding the paper, we only need to do power iteration one time.
    """
    _u = u

    for i in range(rounds):
        _v = _l2normalizer(K.dot(_u, W))
        _u = _l2normalizer(K.dot(_v, K.transpose(W)))

    W_sn = K.sum(K.dot(_u, W) * _v)
    return W_sn, _u, _v


@tf_export('keras.layers.Conv2D', 'keras.layers.Convolution2D')
class Conv2D(tf_convolutional_layers.Conv2D, Layer):
    """2D convolution layer (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.

    Input shape:
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    Output shape:
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 spectral_normalization=True,
                 bias_constraint=None,
                 **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        super(Conv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

        tf.logging.fatal("CONV: Using Spectral Norm!")
        self.u = K.random_normal_variable([1, filters], 0, 1, dtype=self.dtype, name="sn_estimate")  # [1, out_channels]
        self.spectral_normalization = spectral_normalization

    def compute_spectral_normal(self, training=True):
        # Spectrally Normalized Weight
        if self.spectral_normalization:
            # Get kernel tensor shape [kernel_h, kernel_w, in_channels, out_channels]
            W_shape = self.kernel.shape.as_list()

            # Flatten the Tensor
            W_mat = K.reshape(self.kernel, [W_shape[-1], -1])  # [out_channels, N]

            W_sn, u, v = power_iteration(W_mat, self.u)

            if training:
                # Update estimated 1st singular vector
                self.u.assign(u)

            return self.kernel / W_sn
        else:
            return self.kernel

    def call(self, inputs, training=None):

        outputs = K.conv2d(
            inputs,
            self.compute_spectral_normal(training),
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.bias is not None:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.Conv2DTranspose',
           'keras.layers.Convolution2DTranspose')
class Conv2DTranspose(tf_convolutional_layers.Conv2DTranspose, Layer):
    """Transposed convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to the kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.

    Input shape:
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.

    Output shape:
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.

    References:
        - [A guide to convolution arithmetic for deep
          learning](https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional
          Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 spectral_normalization=True,
                 **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        super(Conv2DTranspose, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)
        self.spectral_normalization = spectral_normalization
        self.u = K.random_normal_variable([1, filters], 0, 1, dtype=self.dtype, name="sn_estimate")  # [1, out_channels]
        tf.logging.fatal("DECONV: Using Spectral Norm!")

    def compute_spectral_normal(self, training=True):
        # Spectrally Normalized Weight

        if self.spectral_normalization:
            # Get the kernel tensor shape
            W_shape = self.kernel.shape.as_list()

            # Flatten the Tensor
            # For transpose conv, the kernel shape is [H,W,Out,In]
            W_mat = K.reshape(self.kernel, [W_shape[-2], -1])  # [out_c, N]

            sigma, u, v = power_iteration(W_mat, self.u)

            if training:
                # Update estimated 1st singular vector
                self.u.assign(u)

            return self.kernel / sigma
        else:
            return self.kernel

    # Overwrite the call() method to include Spectral normalization call
    def call(self, inputs, training=True):
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        # Infer the dynamic output shape:
        out_height = utils.deconv_output_length(height,
                                                kernel_h,
                                                self.padding,
                                                stride_h)
        out_width = utils.deconv_output_length(width,
                                               kernel_w,
                                               self.padding,
                                               stride_w)
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
            strides = (1, 1, stride_h, stride_w)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)
            strides = (1, stride_h, stride_w, 1)

        output_shape_tensor = array_ops.stack(output_shape)
        outputs = nn.conv2d_transpose(
            inputs,
            self.compute_spectral_normal(training=training),
            output_shape_tensor,
            strides,
            padding=self.padding.upper(),
            data_format=utils.convert_data_format(self.data_format, ndim=4))

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = inputs.get_shape().as_list()
            out_shape[c_axis] = self.filters
            out_shape[h_axis] = utils.deconv_output_length(out_shape[h_axis],
                                                           kernel_h,
                                                           self.padding,
                                                           stride_h)
            out_shape[w_axis] = utils.deconv_output_length(out_shape[w_axis],
                                                           kernel_w,
                                                           self.padding,
                                                           stride_w)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = nn.bias_add(
                outputs,
                self.bias,
                data_format=utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'spectral_normalization': self.spectral_normalization
        }
        base_config = super(Conv2DTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


Convolution2D = Conv2D
Convolution2DTranspose = Conv2DTranspose
Deconvolution2D = Deconv2D = Conv2DTranspose
'''


import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization,
    Conv2D, Conv2DTranspose,
    add)
from tensorflow.nn import leaky_relu

# pylint: disable=arguments-differ


class Conv(tf.keras.Model):
  def __init__(self, filters, kernel_size, stride=1):
    super(Conv, self).__init__(name="plain_conv_{}_{}/{}".format(filters, kernel_size, stride))
    self.conv = Conv2D(filters, (kernel_size, kernel_size), strides=(stride, stride),
        padding="same", use_bias=False)
  def call(self, x, training=True):
    return self.conv(x)


class Deconv(tf.keras.Model):
  def __init__(self, filters, kernel_size, stride=2):
    super(Deconv, self).__init__(name="plain_deconv_{}_{}/{}".format(filters, kernel_size, stride))
    self.conv = Conv2DTranspose(filters, (kernel_size, kernel_size), strides=(stride, stride),
        padding="same", use_bias=False)
  def call(self, x, training=True):
    return self.conv(x)


class ConvBlock(tf.keras.Model):
  def __init__(self, filters=64, kernel_size=3, stride=1, activation=leaky_relu):
    super(ConvBlock, self).__init__(name="conv_{}_{}/{}".format(filters, kernel_size, stride))
    self.activation = activation
    self.conv = Conv(filters, kernel_size, stride=stride)
    self.norm = BatchNormalization()

  def call(self, x, training=True):
    x = self.conv(x)
    x = self.norm(x, training=training)
    return self.activation(x)


class DeconvBlock(tf.keras.Model):
  def __init__(self, filters=64, kernel_size=3, stride=1, activation=leaky_relu):
    super(DeconvBlock, self).__init__(name="deconv_{}_{}/{}".format(filters, kernel_size, stride))
    self.activation = activation
    self.conv = Deconv(filters, kernel_size, stride=stride)
    self.norm = BatchNormalization()

  def call(self, x, training=True):
    x = self.conv(x)
    x = self.norm(x, training=training)
    return self.activation(x)


class ResizeBlock(tf.keras.Model):
  def __init__(self, target_size, filters=64, kernel_size=3, activation=leaky_relu):
    super(ResizeBlock, self).__init__(name="resize_{}_{}_{}".format("/".join([str(s) for s in target_size]), filters, kernel_size))
    self.target_size = target_size
    self.activation = activation
    self.conv = Conv(filters, kernel_size, stride=1)
    self.norm = BatchNormalization()

  def call(self, x, training=True):
    x = tf.image.resize_nearest_neighbor(x, self.target_size, align_corners=True)
    x = self.conv(x)
    x = self.norm(x, training=training)
    return self.activation(x)


class ResidualBlock(tf.keras.Model):
  # https://blog.waya.ai/deep-residual-learning-9610bb62c355
  def __init__(self, filters=64, kernel_size=3, stride=1, project_shortcut=True, activation=leaky_relu):
    super(ResidualBlock, self).__init__(name="res_{}_{}/{}".format(filters, kernel_size, stride))
    self.activation = activation
    self.conv1 = Conv(filters, kernel_size, stride=stride)
    self.batchnorm1 = BatchNormalization()
    self.conv2 = Conv(filters, kernel_size, stride=1)
    self.batchnorm2 = BatchNormalization()
    self.project_shortcut = project_shortcut or stride != 1
    if self.project_shortcut:
      self.conv_shortcut = Conv(filters, 1, stride=stride)
      self.batchnorm_shortcut = BatchNormalization()

  def call(self, x, training=True):
    shortcut = x
    x = self.conv1(x)
    x = self.batchnorm1(x, training=training)
    x = self.activation(x)
    x = self.conv2(x)
    x = self.batchnorm2(x, training=training)
    if self.project_shortcut:
      shortcut = self.conv_shortcut(shortcut)
      shortcut = self.batchnorm_shortcut(shortcut, training=training)
    x = add([shortcut, x])
    return self.activation(x)


class PreActivationResidualBlock(tf.keras.Model):
  def __init__(self, filters=64, kernel_size=3, stride=1, project_shortcut=True, activation=leaky_relu):
    super(PreActivationResidualBlock, self).__init__(name="preact_res_{}_{}/{}".format(filters, kernel_size, stride))
    self.activation = activation
    self.conv1 = Conv(filters, kernel_size, stride=stride)
    self.batchnorm1 = BatchNormalization()
    self.conv2 = Conv(filters, kernel_size, stride=1)
    self.batchnorm2 = BatchNormalization()
    self.project_shortcut = project_shortcut or stride != 1
    if self.project_shortcut:
      self.conv_shortcut = Conv(filters, 1, stride=stride)
      self.batchnorm_shortcut = BatchNormalization()

  def call(self, x, training=True):
    shortcut = x
    x = self.conv1(x)
    x = self.batchnorm1(x, training=training)
    x = self.activation(x)
    x = self.conv2(x)
    x = self.batchnorm2(x, training=training)
    x = self.activation(x)
    if self.project_shortcut:
      shortcut = self.conv_shortcut(shortcut)
      shortcut = self.batchnorm_shortcut(shortcut, training=training)
    return add([shortcut, x])


class ReverseResidualBlock(tf.keras.Model):
  def __init__(self, filters=64, kernel_size=3, stride=1, project_shortcut=True, activation=leaky_relu):
    super(ReverseResidualBlock, self).__init__(name="revres_{}_{}/{}".format(filters, kernel_size, stride))
    self.activation = activation
    self.conv1 = Conv(filters, kernel_size, stride=1)
    self.batchnorm1 = BatchNormalization()
    self.conv2 = Deconv(filters, kernel_size, stride=stride)
    self.batchnorm2 = BatchNormalization()
    self.project_shortcut = project_shortcut or stride != 1
    if self.project_shortcut:
      self.conv_shortcut = Deconv(filters, 1, stride=stride)
      self.batchnorm_shortcut = BatchNormalization()

  def call(self, x, training=True):
    shortcut = x
    x = self.conv1(x)
    x = self.batchnorm1(x, training=training)
    x = self.activation(x)
    x = self.conv2(x)
    x = self.batchnorm2(x, training=training)
    if self.project_shortcut:
      shortcut = self.conv_shortcut(shortcut)
      shortcut = self.batchnorm_shortcut(shortcut, training=training)
    x = add([shortcut, x])
    return self.activation(x)


class BottleneckResidualBlock(tf.keras.Model):
  # https://blog.waya.ai/deep-residual-learning-9610bb62c355
  def __init__(self, filters=64, kernel_size=3, stride=1, project_shortcut=True, activation=leaky_relu):
    super(BottleneckResidualBlock, self).__init__(name="bres_{}_{}/{}".format(filters, kernel_size, stride))
    self.activation = activation
    self.conv1 = Conv(filters//4, 1, stride=1)
    self.batchnorm1 = BatchNormalization()
    self.conv2 = Conv(filters//4, kernel_size, stride=stride)
    self.batchnorm2 = BatchNormalization()
    self.conv3 = Conv(filters, 1, stride=1)
    self.batchnorm3 = BatchNormalization()
    self.project_shortcut = project_shortcut or stride != 1
    if self.project_shortcut:
      self.conv_shortcut = Conv(filters, 1, stride=stride)
      self.batchnorm_shortcut = BatchNormalization()

  def call(self, x, training=True):
    shortcut = x
    x = self.conv1(x)
    x = self.batchnorm1(x, training=training)
    x = self.activation(x)
    x = self.conv2(x)
    x = self.batchnorm2(x, training=training)
    x = self.activation(x)
    x = self.conv3(x)
    x = self.batchnorm3(x, training=training)
    if self.project_shortcut:
      shortcut = self.conv_shortcut(shortcut)
      shortcut = self.batchnorm_shortcut(shortcut, training=training)
    x = add([shortcut, x])
    return self.activation(x)


class ReverseBottleneckResidualBlock(tf.keras.Model):
  def __init__(self, filters=64, kernel_size=3, stride=1, project_shortcut=True, activation=leaky_relu):
    super(ReverseBottleneckResidualBlock, self).__init__(name="revbres_{}_{}/{}".format(filters, kernel_size, stride))
    self.activation = activation
    self.conv1 = Conv(filters//4, 1, stride=1)
    self.batchnorm1 = BatchNormalization()
    self.conv2 = Deconv(filters//4, kernel_size, stride=stride)
    self.batchnorm2 = BatchNormalization()
    self.conv3 = Deconv(filters, 1, stride=1)
    self.batchnorm3 = BatchNormalization()
    self.project_shortcut = project_shortcut or stride != 1
    if self.project_shortcut:
      self.conv_shortcut = Deconv(filters, 1, stride=stride)
      self.batchnorm_shortcut = BatchNormalization()

  def call(self, x, training=True):
    shortcut = x
    x = self.conv1(x)
    x = self.batchnorm1(x, training=training)
    x = self.activation(x)
    x = self.conv2(x)
    x = self.batchnorm2(x, training=training)
    x = self.activation(x)
    x = self.conv3(x)
    x = self.batchnorm3(x, training=training)
    if self.project_shortcut:
      shortcut = self.conv_shortcut(shortcut)
      shortcut = self.batchnorm_shortcut(shortcut, training=training)
    x = add([shortcut, x])
    return self.activation(x)


class UBlock(tf.keras.Model):
  # NOTE: differences to paper:
  # - skip connection also from input to output
  # - not using arbitrary-looking filter numbers for decoder
  # - not using kernel size 4 and depth 8
  def __init__(self, filters=64, max_filters=512, depth=4, kernel_size=3, activation=leaky_relu):
    super(UBlock, self).__init__(name="u_{}_{}-{}".format(filters, kernel_size, depth))
    self.activation = activation
    self.encoder_convolutions = []
    self.encoder_batchnorms = []
    self.decoder_convolutions = []
    self.decoder_batchnorms = []
    for i in range(depth):
      self.encoder_convolutions.append(Conv(min(filters*2**i, max_filters), kernel_size, stride=2))
      self.encoder_batchnorms.append(BatchNormalization())
      self.decoder_convolutions.append(Deconv(
        min(filters*2**(depth-i-2), max_filters) if i < depth-1 else filters, kernel_size, stride=2))
      self.decoder_batchnorms.append(BatchNormalization())

  def call(self, x, training=True):
    shortcuts = []

    for i in range(len(self.encoder_convolutions)):
      shortcuts.append(x)
      x = self.encoder_convolutions[i](x, training=training)
      x = self.encoder_batchnorms[i](x, training=training)
      x = self.activation(x)

    for i in range(len(self.decoder_convolutions)):
      x = self.decoder_convolutions[i](x, training=training)
      x = self.decoder_batchnorms[i](x, training=training)
      shortcut = shortcuts.pop()
      x = add([shortcut, x])
      x = self.activation(x)

    return x
