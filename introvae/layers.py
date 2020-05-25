"""
Custom Layers for building a Variational Autoencoder

Author: Simon Thomas
Date: 22-May-2020

"""
import numpy as np
import keras.backend as K
from keras.layers import Lambda, BatchNormalization, Add, LeakyReLU, Conv2D,\
                         UpSampling2D, Input, Flatten, Dense, Reshape, Layer, \
                         Cropping2D


def tRGB(x, block):
    """
    A convolutional transformation to RGB space from filter space.
    :param x: the tensor to transform
    :param block: the block number
    :return: rgb
    """
    rgb = Conv2D(3, (1, 1), padding="same", activation="sigmoid", name=f"block_{block}_tRGB")(x)
    return rgb


def fRGB(x, rgb, filters, block):
    """
    A convolutional transformation from RBG space to filter
    space for given image size.
    :param x: the residual tensor
    :param rgb: the image input to transform
    :param filters: the number of filters
    :param block: the block number
    :return: x
    """
    t = Conv2D(filters, (1, 1), strides=(1, 1), padding="same", name=f"block_{block}_fRGB")(rgb)
    t = BatchNormalization(name=f"block_{block}_BN_fRGB")(t)
    t = LeakyReLU(0.2)(t)
    if x is not None:
        x = Conv2D(filters, (3, 3), strides=(1, 1), padding="same", name=f"block_{block}_fRGB_res")(x)
        x = Add()([x, t])
        return x
    return t

def conv_block(x, filters, block, down=True, residual=True):
    """
    Generic convolution block that can perform.

    :param x: tensor to transform
    :param filters: the number of filters
    :param block: the block number
    :param down: upsample convolution or downsample convolution. Residual connections
                    are not included in upsample blocks.
    :param residual: include residual connection
    :return: x
    """
    x_res = x
    if down:
        # DownSample
        # Conv 1
        x = Conv2D(filters, (3, 3), strides=(1, 1), padding="same", name=f"block_{block}_conv1")(x)
        x = BatchNormalization(name=f"block_{block}_BN_1")(x)
        x = LeakyReLU(0.2)(x)

        if residual:
            x = Add(name=f"block_{block}_residual")([x, x_res])
            x = BatchNormalization(name=f"block_{block}_BN_3")(x)
            x = LeakyReLU(0.2)(x)

        # Conv 2
        x = Conv2D(filters, (3, 3), strides=(2, 2), padding="same", name=f"block_{block}_conv2")(x)
        x = BatchNormalization(name=f"block_{block}_BN_2")(x)
        x = LeakyReLU(0.2)(x)

    else:
        # Upsample
        x = UpSampling2D(name=f"block_{block}_upsample")(x)
        x = Conv2D(filters, (4, 4), padding="same", name=f"block_{block}_UpConv2d_1")(x)
        x = BatchNormalization(name=f"block_{block}_BN_1")(x)
        x = LeakyReLU(0.2)(x)
        x = Conv2D(filters, (3, 3), padding="same", name=f"block_{block}_UpConv2d_2")(x)
        x = BatchNormalization(name=f"block_{block}_BN_2")(x)
        x = LeakyReLU(0.2)(x)
    return x

def sampling(args):
    """
    Sampling layer used with the VAE
    """
    mu, log_var = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
    return mu + K.exp(log_var / 2) * epsilon


def add_vae_layer(x, z_dim):
    """
    Adds the variational component, returning the
    z vector plus the mean and standard deviation estimates.

    :param x: the final feature tensor
    :param z_dim: the dimension of the latent space
    :return: z, mu, log_var
    """
    x = Flatten(name="flatten")(x)
    mu = Dense(z_dim, name='mu')(x)
    log_var = Dense(z_dim, name='log_var')(x)
    z = Lambda(sampling, name='z')([mu, log_var])
    return z, mu, log_var

def add_base_layer(z=None, base_dim=(4, 4, 512), style=False):
    """
    Adds the base layer of the generator and returns the lowest
    convolutional feature tensor. All inputs need to be specified.

    :param z: the latent vector z
    :param base_dim: the shape of the lowest convolutional feature e.g. (4x4x512)
    :param style:  indicates whether a style network with constant input
    :return: x OR (constant_input, x)
    """
    if style:
        # Constant Start - 4x4x512
        constant_input = Input(tensor=K.constant([[1]]), name="constant_input")
        x = Dense(np.product(base_dim))(constant_input)
        x = Reshape(base_dim)(x)
        x = BatchNormalization(name="base_BN")(x)
        x = LeakyReLU(0.2)(x)
        return constant_input, x
    else:
        x = Dense(np.prod(base_dim), name="base_flatten")(z)
        x = BatchNormalization(name="base_BN")(x)
        x = LeakyReLU(0.2)(x)
        x = Reshape(base_dim, name="base_Reshape")(x)
        return x

def add_mapping_layers(z, z_dim, n_layers):
    """
    Add a mapping network to the graph
    :param z: the input tensor
    :param z_dim: the dimension of z
    :param n_layers: the number of layers in mapping network
    :return:
    """
    x = z
    for l in range(n_layers):
        x = Dense(z_dim, name=f"mapping_network_{l}")(x)
        x = LeakyReLU(0.2, name=f"mapping_network_relu_{l}")(x)
    return x


def style_block(filters, input_tensor, style_tensor, noise_image, block_num):
    """
    Creates a style block with residual connections.
    :param filters: the number of filters for this block
    :param input_tensor: the tensor to modulate e.g. 4x4x512 -> 8x8x256 and upwards
    :param style_tensor: the style tensor to modulate with i.e. w1, w2, w3 etc.
    :param noise_image: the noise image for this scale e.g. 8x8x1, 16x16x1 etc.
    :param block_num: the depth of the network
    :return: x - the tensor to return
    """
    beta = Dense(filters, name="block_{0}_beta1".format(block_num))(style_tensor)
    beta = Reshape([1, 1, filters], name="block_{0}_beta1_reshape".format(block_num))(beta)
    gamma = Dense(filters, name="block_{0}_gamma1".format(block_num))(style_tensor)
    gamma = Reshape([1, 1, filters], name="block_{0}_gamma1_reshape".format(block_num))(gamma)

    # Learn a bias for the noise at this level
    noise = Conv2D(filters=filters,
                   kernel_size=1,
                   padding='same',
                   kernel_initializer='he_normal',
                   name="block_{0}_noise_bias1".format(block_num)
                   )(noise_image)

    # Phase 1
    x = UpSampling2D(name="block_{0}_UpSample".format(block_num))(input_tensor)
    x = Conv2D(filters=filters,
               kernel_size=3,
               padding='same',
               kernel_initializer='he_normal',
               name="block_{0}_decoder_conv1".format(block_num)
               )(x)
    x = AdaInstanceNormalization(name="block_{0}_AdaIN_1".format(block_num))([x, beta, gamma])
    x = Add()([x, noise])
    x = LeakyReLU(name="block_{0}_decoder_LRelu1".format(block_num))(x)

    # Phase 2
    beta = Dense(filters, name="block_{0}_beta2".format(block_num))(style_tensor)
    beta = Reshape([1, 1, filters], name="block_{0}_beta2_reshape".format(block_num))(beta)
    gamma = Dense(filters, name="block_{0}_gamma2".format(block_num))(style_tensor)
    gamma = Reshape([1, 1, filters], name="block_{0}_gamma2_reshape".format(block_num))(beta)

    # Learn a bias for the noise at this level
    noise = Conv2D(filters=filters,
                   kernel_size=1,
                   padding='same',
                   kernel_initializer='he_normal',
                   name="block_{0}_noise_bias2".format(block_num)
                   )(noise_image)

    x = Conv2D(filters=filters,
               kernel_size=3,
               padding='same',
               kernel_initializer='he_normal',
               name="block_{0}_decoder_conv2".format(block_num)
               )(x)
    x = AdaInstanceNormalization(name="block_{0}_AdaIN_2".format(block_num))([x, beta, gamma])
    x = Add()([x, noise])
    x = LeakyReLU(name="block_{0}_decoder_LRelu2".format(block_num))(x)

    return x



class BilinearUpSample(Lambda):
    """
    Bilinear UpSamplling Layer.

    Input:
        name - the name of the layer
        **kwargs - keyword arguments
    """
    def __init__(self, name='bilinear_upsample', **kwargs):
        # Function
        func = (Lambda(K.resize_images,
                      arguments={"height_factor": 2,
                                 "width_factor": 2,
                                 "data_format": "channels_last",
                                 "interpolation": "bilinear"
                                }))
        super(BilinearUpSample, self).__init__(func, name=name)

    def get_config(self):
        """Return the config of the layer."""
        config = super(BilinearUpSample, self).get_config()
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create a layer from its config."""
        return cls(**config, custom_objects={'bilinear_upsample': BilinearUpSample})

    def __repr__(self):
        return f"<Keras CustomLayer - BiLinearUpSample - {self.name}>"

class BilinearDownSample(Lambda):
    """
    Bilinear DownSamplling Layer.

    Input:
        name - the name of the layer
        **kwargs - keyword arguments
    """
    def __init__(self, name='bilinear_downsample', **kwargs):
        # Function
        func = (Lambda(K.resize_images,
                      arguments={"height_factor": 0.5,
                                 "width_factor": 0.5,
                                 "data_format": "channels_last",
                                 "interpolation": "bilinear"
                                }))
        super(BilinearDownSample, self).__init__(func, name=name)

    def get_config(self):
        """Return the config of the layer."""
        config = super(BilinearDownSample, self).get_config()
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create a layer from its config."""
        return cls(**config, custom_objects={'bilinear_downsample': BilinearDownSample})

    def __repr__(self):
        return f"<Keras CustomLayer - BiLinearDownSample - {self.name}>"


class AdaInstanceNormalization(Layer):
    """
    This is the AdaInstanceNormalization layer used by

    manicman199 available at https://github.com/manicman1999/StyleGAN-Keras

    """

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):

        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')

        super(AdaInstanceNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))

        beta = inputs[1]
        gamma = inputs[2]

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):

        return input_shape[0]

