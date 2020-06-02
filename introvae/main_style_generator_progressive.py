"""
Example of how to do it...

TO DO

"""

from introvae.models import *
from introvae.data import DataGen

import tensorflow as tf
import matplotlib.pyplot as plt

import skimage.io as io

# Set Seed
np.random.seed(0)

# MODEL SETTINGS
IMG_DIM = 64
Z_DIM = 100
FILTERS = [16, 32, 64, 128]
BASE_DIM = (4, 4, 128)
N_LAYERS = len(FILTERS)
N_MAPPING_LAYERS = 4
BATCH_SIZE = 64
IMG_DIR = "/home/simon/Documents/Programming/Data/progressive_growing_of_gans/celeba-hq/celeba-64/"
LEARNING_RATE = 0.0002

# Data
gen = DataGen(IMG_DIR, IMG_DIM, BATCH_SIZE, style=True)

((X, noise), y) = gen[0]

# Models
tensors, encoder = build_encoder(IMG_DIM, Z_DIM, FILTERS, True)
z, z_mean, z_log_var = tensors

generator = build_style_generator(IMG_DIM, Z_DIM, FILTERS, BASE_DIM, N_MAPPING_LAYERS)
#generator = build_generator(Z_DIM, FILTERS, BASE_DIM)


# Losses
from keras.objectives import mean_squared_error


def reg_loss(mean, log_var):
    return K.mean(0.5 * K.sum(- 1 - log_var + K.square(mean) + K.exp(log_var), axis=-1))


def mse_loss(x, x_decoded):
    original_dim = np.float32(np.prod((IMG_DIM, IMG_DIM, 3)))
    return K.mean(original_dim * mean_squared_error(x, x_decoded))


generator_input_z = generator.inputs[0]
generator_input_noise = generator.inputs[1]
generator_input_constant = generator.inputs[2]
encoder_input = encoder.get_input_at(0)

# feed_dict tensors
xr = generator([z, generator_input_noise, generator_input_constant])
reconst_latent_input = Input(batch_shape=(BATCH_SIZE, Z_DIM))
_, zr_mean, zr_log_var = encoder(generator([reconst_latent_input, generator_input_noise, generator_input_constant]))
_, zr_mean_ng, zr_log_var_ng = encoder(K.stop_gradient(generator([reconst_latent_input, generator_input_noise, generator_input_constant])))
xr_latent = generator([reconst_latent_input, generator_input_noise, generator_input_constant])

sampled_latent_input = Input(batch_shape=(BATCH_SIZE, Z_DIM), name='sampled_latent_input')
_, zpp_mean, zpp_log_var = encoder(generator([sampled_latent_input, generator_input_noise, generator_input_constant]))
_, zpp_mean_ng, zpp_log_var_ng = encoder(K.stop_gradient(generator([sampled_latent_input, generator_input_noise, generator_input_constant])))


# KL
l_reg_z = reg_loss(z_mean, z_log_var)
l_reg_zr_ng = reg_loss(zr_mean_ng, zr_log_var_ng)
l_reg_zpp_ng = reg_loss(zpp_mean_ng, zpp_log_var_ng)

# Reconstruction
l_ae = mse_loss(encoder_input, xr)
l_ae2 = mse_loss(encoder_input, xr_latent)

ALPHA = 0.25
BETA = 0.1
DELTA = 1
M = 18


# ENCODER LOSSES
encoder_l_adv = DELTA*l_reg_z + ALPHA * K.maximum(0., M - l_reg_zr_ng) + ALPHA * K.maximum(0., M - l_reg_zpp_ng)
encoder_loss = encoder_l_adv + BETA * l_ae

# KL
l_reg_zr = reg_loss(zr_mean, zr_log_var)
l_reg_zpp = reg_loss(zpp_mean, zpp_log_var)

# GENERATOR LOSSES
generator_l_adv = ALPHA * l_reg_zr + ALPHA * l_reg_zpp
generator_loss = generator_l_adv + BETA * l_ae2

# Training --------------------------------------- #

encoder_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
generator_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

encoder_params = encoder.trainable_weights
generator_params = generator.trainable_weights

encoder_grads = encoder_optimizer.compute_gradients(encoder_loss, var_list=encoder_params)
encoder_apply_grads_op = encoder_optimizer.apply_gradients(encoder_grads)

generator_grads = generator_optimizer.compute_gradients(generator_loss, var_list=generator_params)
generator_apply_grads_op = generator_optimizer.apply_gradients(generator_grads)


# START
session = K.get_session()
init = tf.global_variables_initializer()
session.run([init])

generator.load_weights("./weights/generator_64x64_face_residual_304.h5")
encoder.load_weights("./weights/encoder_64x64_face_residual_304.h5")

epochs = 500

losses = {
    "enc_loss_np": [],
    "enc_l_ae_np": [],
    "l_reg_z_np": [],
    "l_reg_zr_ng_np": [],
    "l_reg_zpp_ng_np": [],
    "generator_loss_np": [],
    "dec_l_ae_np": [],
    "l_reg_zr_np": [],
    "l_reg_zpp_np": []
}


# test data
z_test = np.random.uniform(0, 1, (BATCH_SIZE, Z_DIM))

start = 305
for epoch in range(start, epochs):

    iterations = gen.n // gen.batch_size
    for i in range(iterations):

        (x, noise), _ = gen[i]

        z_p = np.random.normal(loc=0.0, scale=1.0, size=(BATCH_SIZE, Z_DIM))
        z_x, x_r, x_p = session.run([z, xr, generator.get_output_at(0)],
                                    feed_dict={encoder_input: x,
                                               generator_input_z: z_p,
                                               generator_input_noise: noise
                                               })

        # Train encoder
        _ = session.run([encoder_apply_grads_op],
                        feed_dict={encoder_input: x,
                                   reconst_latent_input: z_x,
                                   sampled_latent_input: z_p,
                                   generator_input_noise: noise
                                   })
        # Train generator
        _ = session.run([generator_apply_grads_op],
                        feed_dict={encoder_input: x,
                                   reconst_latent_input: z_x,
                                   sampled_latent_input: z_p,
                                   generator_input_noise: noise
                                   })

        if (i % 100) == 0:
            enc_loss_np, \
            enc_l_ae_np, \
            l_reg_z_np, \
            l_reg_zr_ng_np, \
            l_reg_zpp_ng_np, \
            generator_loss_np, \
            dec_l_ae_np, \
            l_reg_zr_np, \
            l_reg_zpp_np = \
                session.run([encoder_loss, l_ae, l_reg_z,
                             l_reg_zr_ng, l_reg_zpp_ng, generator_loss,
                             l_ae2, l_reg_zr, l_reg_zpp],
                            feed_dict={encoder_input: X,
                                       reconst_latent_input: z_x,
                                       sampled_latent_input: z_p,
                                       generator_input_noise: noise
                                       })

            print('Epoch: {}/{}, iteration: {}/{}'.format(epoch + 1, epochs, i + 1, iterations))
            print(' Enc_loss: {}, l_ae:{},  l_reg_z: {}, l_reg_zr_ng: {}, l_reg_zpp_ng: {}'.format(
                enc_loss_np, enc_l_ae_np, l_reg_z_np, l_reg_zr_ng_np, l_reg_zpp_ng_np))

            print(' Dec_loss: {}, l_ae:{}, l_reg_zr: {}, l_reg_zpp: {}'.format(
                generator_loss_np, dec_l_ae_np, l_reg_zr_np, l_reg_zpp_np))

            #losses["enc_loss_np"].append(enc_loss_np)
            #losses["generator_loss_np"].append(generator_loss_np)
            losses["dec_l_ae_np"].append(dec_l_ae_np)
            losses["enc_l_ae_np"].append(enc_l_ae_np)
            losses["l_reg_z_np"].append(l_reg_z_np)
            losses["l_reg_zr_ng_np"].append(l_reg_zr_ng_np)
            losses["l_reg_zpp_ng_np"].append(l_reg_zpp_ng_np)
            losses["l_reg_zr_np"].append(l_reg_zr_np)
            losses["l_reg_zpp_np"].append(l_reg_zpp_np)

    # End of epoch
    # data_gen.alpha += 0.1

    # show progress
    N_TO_SHOW = 12
    imgs = [img for img in x][0:N_TO_SHOW]
    img1 = np.hstack(imgs)

    # Reconstruction
    zs, _, _ = encoder.predict(x)
    imgs = [img for img in generator.predict([zs, noise])][0:N_TO_SHOW]
    img2 = np.hstack(imgs)

    # Sample
    imgs = [img for img in generator.predict([z_test, noise])][0:N_TO_SHOW]
    img3 = np.hstack(imgs)

    canvas = np.vstack([img1, img2, img3])

    io.imsave("./img_out/progress_{0:04d}.png".format(epoch), np.clip(canvas, 0, 1))

    for key in losses:
        if key == "dec_l_ae_np" or key == "enc_l_ae_np":
            continue
        plt.plot(range(len(losses[key]) - 1), losses[key][1:], label=key)

    plt.hlines(M, 0, len(losses[key]) - 1, linestyle="--", color="black")
    plt.legend()
    plt.savefig("./img_out/progress_graph.png", dpi=100)
    plt.close()

    print("\n\n")

    generator.save_weights(f"./weights/generator_64x64_face_residual_{epoch:03d}.h5")
    encoder.save_weights(f"./weights/encoder_64x64_face_residual_{epoch:03d}.h5")

