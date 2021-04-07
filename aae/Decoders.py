from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Conv2DTranspose, Dropout, Activation, Flatten, \
    Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate, Conv2DTranspose, GlobalAveragePooling2D, ReLU, add
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from params import PARAMS


def decoder_res_net_v0(z_size, volume_size):
    encoded = Input(shape=(z_size,))
    y = Dense(np.prod(volume_size[1:]), name='encoder_input', kernel_initializer='normal')(encoded)
    b4_out = Reshape((volume_size[1], volume_size[2], volume_size[3]))(y)

    '''block 4'''

    # #b4_avg_p = GlobalAveragePooling2D()(y)
    # b4_out = Dense(z_size, name='model_output', activation='softmax',
    #                kernel_initializer='he_uniform')(y)

    '''block 3'''
    b3_cnv2d_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        use_bias=False, name='b3_cnv2d_1', kernel_initializer='normal')(b4_out)
    b3_relu_1 = ReLU(name='b3_relu_1')(b3_cnv2d_1)
    b3_bn_1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b3_bn_1')(b3_relu_1)  # size: 4*4

    b3_add = add([b4_out, b3_bn_1])  #

    b3_cnv2d_2 = UpSampling2D((2, 2))(b3_add)
    b3_cnv2d_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                        use_bias=False, name='b3_cnv2d_2', kernel_initializer='normal')(b3_cnv2d_2)
    b3_relu_2 = ReLU(name='b3_relu_2')(b3_cnv2d_2)
    b3_out = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b3_out')(b3_relu_2)  # size: 8*8

    '''block 2'''
    b2_cnv2d_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        use_bias=False, name='b2_cnv2d_1', kernel_initializer='normal')(b3_out)
    b2_relu_1 = ReLU(name='b2_relu_1')(b2_cnv2d_1)
    # b2_bn_1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b2_bn_1')(b2_relu_1)  # size: 8*8

    b2_add = add([b3_out, b2_relu_1])  #

    b2_cnv2d_2 = UpSampling2D((2, 2))(b2_add)
    b2_cnv2d_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                        use_bias=False, name='b2_cnv2d_2', kernel_initializer='normal')(b2_cnv2d_2)
    b2_relu_2 = ReLU(name='b2_relu_2')(b2_cnv2d_2)
    # b2_out = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b2_bn_2')(b2_relu_2)  # size: 16*16

    '''block_1'''
    b1_cnv2d_1 = UpSampling2D((2, 2))(b2_relu_2)
    b1_cnv2d_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same',
                        use_bias=False, name='b1_cnv2d_1', kernel_initializer='normal')(b1_cnv2d_1)
    b1_relu_1 = ReLU(name='b1_relu_1')(b1_cnv2d_1)
    # b1_bn_1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b1_bn_1')(b1_relu_1)  # size: 32*32

    b1_cnv2d_2 = UpSampling2D((2, 2))(b1_relu_1)
    b1_cnv2d_2 = Conv2D(filters=16, kernel_size=(1, 1), padding='same',
                        use_bias=False, name='b1_cnv2d_2', kernel_initializer='normal')(b1_cnv2d_2)
    b1_relu_2 = ReLU(name='b1_relu_2')(b1_cnv2d_2)
    # b1_out = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b1_out')(b1_relu_2)  # size: 64*64

    reconstruction = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same')(b1_relu_2)
    model = Model(encoded, reconstruction)

    # model.summary()
    return model


def make_decoder_model_paper(z_size, volume_size):
    filters = 3
    # start building the decoder model which will accept the
    # output of the encoder as its inputs
    encoded = tf.keras.layers.Input(shape=(z_size,))
    y = Dense(np.prod(volume_size[1:]))(encoded)
    y = Reshape((volume_size[1], volume_size[2], volume_size[3]))(y)

    y = Conv2D(filters=filters * 100, kernel_size=3, activation=PARAMS['activation'], padding="same")(y)
    y = UpSampling2D((2, 2))(y)

    y = Conv2D(filters=filters * 60, kernel_size=4, activation=PARAMS['activation'], padding="same")(y)
    y = UpSampling2D((2, 2))(y)
    # ###
    # y = Conv2D(filters=filters*40, kernel_size=4, activation="relu", padding="same")(y)
    # y = UpSampling2D((2, 2))(y)
    # ###
    y = Conv2D(filters=filters * 20, kernel_size=5, activation=PARAMS['activation'], padding="same")(y)
    y = UpSampling2D((2, 2))(y)
    decoded = Conv2D(1, (5, 5), activation=PARAMS['activation'], padding="same")(y)

    ####
    # reconstruction = tf.keras.layers.Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(y)
    decoder = tf.keras.Model(inputs=encoded, outputs=decoded)
    decoder.summary()
    return decoder


def my_make_decoder_model(z_size, volume_size):
    filters = 64
    # start building the decoder model which will accept the
    # output of the encoder as its inputs
    encoded = tf.keras.layers.Input(shape=(z_size,))
    y = Dense(np.prod(volume_size[1:]))(encoded)
    y = Reshape((volume_size[1], volume_size[2], volume_size[3]))(y)

    y = Conv2D(filters * 8, (3, 3), activation="relu", padding="same")(y)
    y = UpSampling2D((2, 2))(y)

    y = Conv2D(filters * 4, (3, 3), activation="relu", padding="same")(y)
    y = UpSampling2D((2, 2))(y)

    y = Conv2D(filters * 2, (3, 3), activation="relu", padding="same")(y)
    y = UpSampling2D((2, 2))(y)

    y = Conv2D(filters * 1, (5, 5), activation="relu", padding="same")(y)

    y = UpSampling2D((2, 2))(y)
    reconstruction = Conv2D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(y)
    decoder = Model(inputs=encoded, outputs=reconstruction)
    # decoder.summary()
    return decoder


def make_decoder_CAE(z_size, volume_size):
    encoded = Input(shape=(z_size,))
    y = tf.keras.layers.Dense(np.prod(volume_size[1:]))(encoded)
    y = tf.keras.layers.Reshape((volume_size[1], volume_size[2], volume_size[3]))(y)

    y = Conv2DTranspose(filters=100, kernel_size=4, strides=(2, 2), padding="same")(y)
    y = Conv2DTranspose(filters=128, kernel_size=4, strides=(2, 2), padding="same")(y)
    y = Conv2DTranspose(filters=64, kernel_size=4, strides=(2, 2), padding="same")(y)
    y = Conv2DTranspose(filters=32, kernel_size=4, strides=(2, 2), padding="same")(y)
    y = Conv2DTranspose(filters=16, kernel_size=4, strides=(2, 2), padding="same")(y)

    reconstruction = Conv2DTranspose(filters=1, kernel_size=4, activation='relu', padding='same')(y)
    model = tf.keras.Model(inputs=encoded, outputs=reconstruction)
    model.summary()
    return model


def make_decoder_vae_ass1(z_size, volume_size):
    latent_inputs = Input(shape=(z_size,), name='z_sampling')
    x = Dense(volume_size[1] * volume_size[2] * volume_size[3], activation='relu')(latent_inputs)
    x = Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)
    x = Conv2DTranspose(32, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(16, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(16, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    dec_output = Conv2DTranspose(1, kernel_size=3, activation='relu', padding='same')(x)
    decoder = Model(latent_inputs, dec_output, name='origin_decoder')
    return decoder
