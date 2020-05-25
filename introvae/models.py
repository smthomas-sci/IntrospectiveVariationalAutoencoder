"""

An Introspective Variational Autoencoder module.

Author: Simon Thomas
Date: 22-May-2020

"""
from introvae.layers import *
from keras.models import Model
import tensorflow as tf

def build_encoder(img_dim, z_dim, filters, residual=True):
    """
    Builds generic encoder and returns tensors to create the loss

    :param img_dim: the input image size
    :param z_dim: the size of the latent dim
    :param filters: a list of filters for each block
    :param residual: include residual connections
    :return: tensors[z, mu, log_var], model
    """
    n_layers = len(filters)
    encoder_input = Input(shape=(img_dim, img_dim, 3), name="encoder_input")

    x = None
    rgb = encoder_input
    for i in range(n_layers):
        # Map RGB
        x = fRGB(x, rgb, filters[i], block=i)
        # Add Standard Residual Convolution
        x = conv_block(x, filters[i], block=i, down=True, residual=residual)
        # Downsample RGB
        rgb = BilinearDownSample(name=f"block_{i}_Bilinear_Down")(rgb)

    # VAE Component
    z, mu, log_var = add_vae_layer(x, z_dim)
    tensors = [z, mu, log_var]
    # Build Model
    model = Model(inputs=[encoder_input], outputs=[z, mu, log_var], name="encoder")

    return tensors, model

def build_style_generator(img_dim, z_dim, filters, base_dim, n_mapping_layers):
    """
    Builds a style-based generator.
    :param img_dim: the input image size
    :param z_dim: the size of the latent dim
    :param filters: a list of filters for each block
    :param base_dim: the shape of the base feature tensor
    :param n_mapping_layers: the number of layers in the mapping network
    :return: model - the generator model
    """
    n_layers = len(filters)
    generator_input = Input(shape=(z_dim,), name="generator_input")
    z = generator_input

    # Crop the noise images
    noise_input = Input(shape=(img_dim, img_dim, 1), name="noise_image")
    noise_img = noise_input
    noise_images = [noise_img]
    curr_size = img_dim
    while curr_size > 8:
        curr_size = int(curr_size / 2)
        cut = int(curr_size / 2)
        crop = Cropping2D(cut)(noise_img)
        noise_img = crop
        noise_images.insert(0, crop)

    w = add_mapping_layers(z, z_dim, n_mapping_layers)
    constant_input, x = add_base_layer(z=False, base_dim=base_dim, style=True)

    rgbs = []
    for i in range(n_layers):
        rgb = tRGB(x, block=i)
        x = style_block(filters[::-1][i], x, w, noise_images[i], i)
        rgbs.append(rgb)

    # Get final RGB and then sum
    rgb = tRGB(x, block="final")
    rgbs.append(rgb)

    # Sum the RGBs
    img = rgbs[0]
    SUM = Add(name="Sum")
    BUS =BilinearUpSample(name=f"bilinear_upsample")
    for i, rgb in enumerate(rgbs[1::]):
        img = BUS(img)
        img = SUM([img, rgb])

    generator_output = img
    model = Model(inputs=[generator_input, noise_input, constant_input], outputs=[generator_output],
                  name="generator_style")

    return model

def build_generator(z_dim, filters, base_dim):
    """
    Builds an RGB generator.
    :param z_dim: the size of the latent dim
    :param filters: a list of filters for each block
    :param base_dim: the shape of the base feature tensor
    :return: model - the generator model
    """
    n_layers = len(filters)
    generator_input = Input(shape=(z_dim,), name="generator_input")
    z = generator_input

    x = add_base_layer(z=z, base_dim=base_dim, style=False)

    rgbs = []
    for i in range(n_layers):
        rgb = tRGB(x, block=i)
        x = conv_block(x, filters[::-1][i], block=i, down=False, residual=False)
        rgbs.append(rgb)

    # Get final RGB and then sum
    rgb = tRGB(x, block="final")
    rgbs.append(rgb)

    # Sum the RGBs
    img = rgbs[0]
    SUM = Add(name="Sum")
    BUS = BilinearUpSample(name=f"bilinear_upsample")
    for i, rgb in enumerate(rgbs[1::]):
        img = BUS(img)
        img = SUM([img, rgb])

    generator_output = img
    model = Model(inputs=[generator_input], outputs=[generator_output], name="generator")

    return model


# --------- TEST ---------- #
if __name__ == "__main__":
    # MODEL SETTINGS
    IMG_DIM = 128
    Z_DIM = 100
    FILTERS = [16, 32, 64, 128, 256]
    BASE_DIM = (4, 4, 256)
    N_LAYERS = len(FILTERS)
    N_MAPPING_LAYERS = 5


    tensors, encoder = build_encoder(IMG_DIM, Z_DIM, FILTERS, True)
    z, mu, log_var = tensors
    encoder.summary()

    style_generator = build_style_generator(IMG_DIM, Z_DIM, FILTERS, BASE_DIM, N_MAPPING_LAYERS)

    style_generator.summary()

    generator = build_generator(Z_DIM, FILTERS, BASE_DIM)

    generator.summary()




