"""
Custom Layers for building a Variational Autoencoder

Author: Simon Thomas
DateL 22-May-2020

"""
import keras.backend as K
from keras.layers import Lambda, BatchNormalization, Add, LeakyReLU, Conv2D,\
                         UpSampling2D, Input, Flatten, Dense, Reshape


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
    return t if x is None else x + t

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

def add_base_layer(z, base_dim):
    """
    Adds the base layer of the generator and returns the lowest
    convolutional feature tensor.

    :param z: the latent vector z
    :param base_dim: the shape of the lowest convolutional feature e.g. (4x4x512)
    :return: x
    """
    x = Dense(K.prod(base_dim), name="base_flatten")(z)
    x = BatchNormalization(name="base_BN")(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape(base_dim, name="base_Reshape")(x)
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



