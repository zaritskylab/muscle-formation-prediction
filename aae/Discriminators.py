from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Conv2DTranspose, Dropout, Activation, Flatten, \
    Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate, LeakyReLU
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import ELU




def make_discriminator_model(z_size):
    encoded = Input(shape=(z_size,))
    x = Dense(1000)(encoded)
    x = LeakyReLU(0.2)(x)

    x = Dense(1000)(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(1000)(x)
    x = LeakyReLU(0.2)(x)

    prediction = Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=encoded, outputs=prediction)
    # model.summary()
    return model


def create_discriminator(latent_dim):
    input_layer = Input(shape=(latent_dim,))
    disc = Dense(128)(input_layer)
    disc = ELU()(disc)
    disc = Dense(64)(disc)
    disc = ELU()(disc)
    disc = Dense(1, activation="sigmoid")(disc)

    model = tf.keras.Model(input_layer, disc)
    return model


def make_discriminator_model_con(z_size):
    pass
