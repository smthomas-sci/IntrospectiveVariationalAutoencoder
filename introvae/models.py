"""

An Introspective Variational Autoencoder module.

Author: Simon Thomas
Date: 22-May-2020

"""
from introvae.layers import *
from keras.models import Model
import tensorflow as tf

# MODEL SETTINGS
IMG_DIM = 128
Z_DIM = 100
FILTERS = [16, 32, 64, 128]
BASE_DIM = (4, 4, 128)
N_LAYERS = len(FILTERS)

# Build Encoder
encoder_input = Input(shape=(IMG_DIM, IMG_DIM, 3), name="encoder_input")

x = None
rgb = encoder_input
for i in range(N_LAYERS):
    # Map RGB
    x = fRGB(x, rgb, FILTERS[i], block=i)
    # Add Standard Residual Convolution
    x = conv_block(x, FILTERS[i], block=i, down=True, residual=True)
    # Downsample RGB
    rgb = BilinearDownSample(name=f"block_{i}_Bilinear_Down")(rgb)

# VAE Component
z, mu, log_var = add_vae_layer(x, Z_DIM)

# Build Model
encoder = Model(inputs=[encoder_input],
                outputs=[z, mu, log_var],
                name="encoder")

encoder.summary()
# ------------------------------------------------------------------------ #

# Build Decoder
generator_input = Input(shape=(Z_DIM,), name="generator_input")

rgbs = []

x = add_base_layer(generator_input, BASE_DIM)

for i in range(N_LAYERS):
    rgb = tRGB(x, block=0)
    x = conv_block(x, FILTERS[::-1][i], block=i, down=False, residual=False)
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
generator = Model(inputs=[generator_input], outputs=[generator_output], name="generator")

generator.summary()





