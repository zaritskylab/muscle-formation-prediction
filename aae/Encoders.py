from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Conv2DTranspose, Dropout, Activation, Flatten, \
    Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate, Convolution2D, LeakyReLU, add, Lambda, MaxPool2D
import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.python.keras.layers import ReLU, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from params import PARAMS
from tensorflow.keras import backend as K


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def sample_res_net_v0(img_size, z_size):
    """
    :param input_shape: [64, 64, 1]
    :param output_shape: 10
    :return:
    """
    input = Input(shape=(img_size, img_size, 1))

    '''block_1'''
    b1_cnv2d_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False, name='b1_cnv2d_1', kernel_initializer='normal')(input)
    b1_relu_1 = ReLU(name='b1_relu_1')(b1_cnv2d_1)
    b1_bn_1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b1_bn_1')(b1_relu_1)  # size: 32*32

    b1_cnv2d_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same',
                        use_bias=False, name='b1_cnv2d_2', kernel_initializer='normal')(b1_bn_1)
    b1_relu_2 = ReLU(name='b1_relu_2')(b1_cnv2d_2)
    b1_out = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b1_out')(b1_relu_2)  # size: 16*16

    '''block 2'''
    b2_cnv2d_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        use_bias=False, name='b2_cnv2d_1', kernel_initializer='normal')(b1_out)
    b2_relu_1 = ReLU(name='b2_relu_1')(b2_cnv2d_1)
    b2_bn_1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b2_bn_1')(b2_relu_1)  # size: 16*16

    b2_add = add([b1_out, b2_bn_1])  #

    b2_cnv2d_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False, name='b2_cnv2d_2', kernel_initializer='normal')(b2_add)
    b2_relu_2 = ReLU(name='b2_relu_2')(b2_cnv2d_2)
    b2_out = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b2_bn_2')(b2_relu_2)  # size: 8*8

    '''block 3'''
    b3_cnv2d_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        use_bias=False, name='b3_cnv2d_1', kernel_initializer='normal')(b2_out)
    b3_relu_1 = ReLU(name='b3_relu_1')(b3_cnv2d_1)
    b3_bn_1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b3_bn_1')(b3_relu_1)  # size: 8*8

    b3_add = add([b2_out, b3_bn_1])  #

    b3_cnv2d_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False, name='b3_cnv2d_2', kernel_initializer='normal')(b3_add)
    b3_relu_2 = ReLU(name='b3_relu_2')(b3_cnv2d_2)
    b3_out = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b3_out')(b3_relu_2)  # size: 4*4

    volume_size = tf.keras.backend.int_shape(b3_out)
    '''block 4'''
    b4_avg_p = GlobalAveragePooling2D()(b3_out)
    flattened = Flatten()(b4_avg_p)
    output = Dense(z_size, name='model_output', activation='softmax',
                   kernel_initializer='he_uniform')(flattened)  # softmax

    z_mean = Dense(z_size, name='z_mean')(flattened)
    z_log_var = Dense(z_size, name='z_log_var')(flattened)
    z = Lambda(sampling, output_shape=(z_size,), name='z')([z_mean, z_log_var])
    encoder_vae = Model(input, [z_mean, z_log_var, z], name='vae_encoder')

    model = Model(input, output)

    # model_json = model.to_json()
    # with open("sample_res_net_v0.json", "w") as json_file:
    #     json_file.write(model_json)
    # model.summary()
    return model, volume_size, encoder_vae, z_mean, z_log_var


def make_encoder_model_paper(z_size, img_size):
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 1))
    filters = 3
    x = Conv2D(filters=filters * 20, kernel_size=5, activation=PARAMS['activation'], padding="same")(inputs)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    # ###
    # x = Conv2D(filters=filters * 40, kernel_size=5, activation="relu", padding="same")(x)
    # x = MaxPooling2D((2, 2), padding="same")(x)
    # x = BatchNormalization()(x)
    # ###
    x = Conv2D(filters=filters * 60, kernel_size=4, activation=PARAMS['activation'], padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=filters * 100, kernel_size=3, activation=PARAMS['activation'], padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = BatchNormalization()(x)

    x = Dense(PARAMS['dense_units'])(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(z_size)(x)
    x = Dropout(rate=0.2)(x)
    x = Concatenate(axis=-1)([x, x])

    # flatten the network and then construct our latent vector
    # global volume_size
    volume_size = tf.keras.backend.int_shape(x)
    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(z_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=z, name="encoder")
    model.summary()
    return model, volume_size


def my_make_encoder_model(z_size, img_size):
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 1))
    filters = 64

    x = tf.keras.layers.Conv2D(filters * 1, (5, 5), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters * 2, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters * 4, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    ###
    x = tf.keras.layers.Conv2D(filters * 8, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    ###
    # flatten the network and then construct our latent vector
    # global volume_size
    volume_size = tf.keras.backend.int_shape(x)
    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(z_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=z, name="encoder")
    # model.summary()
    return model, volume_size


def make_encoder_CAE(z_size, img_size):
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 1))

    x = Conv2D(filters=16, kernel_size=4, strides=(2, 2), padding="same")(inputs)
    x = Conv2D(filters=32, kernel_size=4, strides=(2, 2), padding="same")(x)
    x = Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding="same")(x)
    x = Conv2D(filters=128, kernel_size=4, strides=(2, 2), padding="same")(x)
    x = Conv2D(filters=100, kernel_size=4, strides=(2, 2), padding="same")(x)

    # flatten the network and then construct our latent vector
    # global volume_size
    volume_size = tf.keras.backend.int_shape(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(z_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="encoder")
    model.summary()
    return model, volume_size


def make_encoder_vae_ass1(z_size, img_size):
    input_img = Input(shape=(img_size, img_size, 1), )

    x = Conv2D(32, (3, 3), activation=PARAMS['activation'], padding='same', kernel_initializer='he_uniform')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation=PARAMS['activation'], padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation=PARAMS['activation'], padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation=PARAMS['activation'], padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), activation=PARAMS['activation'], kernel_initializer='he_uniform', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation=PARAMS['activation'], kernel_initializer='he_uniform', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Dropout(0.4)(x)

    shape = K.int_shape(x)

    x = Flatten()(x)
    x = Dense(PARAMS['dense_units'], activation=PARAMS['activation'], kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='softmax')(x)

    z_mean = Dense(z_size, name='z_mean')(x)
    z_log_var = Dense(z_size, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(z_size,), name='z')([z_mean, z_log_var])

    # encoder = Model(input_img, [z_mean, z_log_var, z], name='origin_encoder')

    return z, z_mean, z_log_var, shape
