from os import listdir
from os import path
import neptune
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Lambda, Reshape, Conv2DTranspose, BatchNormalization, MaxPool2D, \
    Dropout, Dense, Input, ReLU, add, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.python.keras.callbacks import LambdaCallback, EarlyStopping, LearningRateScheduler

from NeptuneCallback import NeptuneCallback
from Trainers import log_data
from datagen import generate_Data, val_generator, aae_load_images
from params import PARAMS, api_token

from NeptuneMethods import lr_scheduler
import numpy as np

latent_dim = PARAMS['z_dim']
image_size = 64
directory = "../../data/images/train"
if not path.exists(directory):
    directory = "muscle-formation-diff/data/images/train"
batch_size = PARAMS['batch_size']
steps = len(listdir(directory)) // batch_size
x_test = aae_load_images(image_size=image_size)[:30]


def block_A(in_b):
    # block A with conv (stride = 1X1)
    b = BatchNormalization()(in_b)
    b = ReLU()(b)
    b = Conv2D(64, (3, 3), padding='same', strides=(1, 1))(b)
    b = BatchNormalization()(b)
    b = ReLU()(b)
    out_b = Conv2D(64, (3, 3), padding='same', strides=(1, 1))(b)
    return out_b


def inverse_block_A(in_b):
    # block A with conv (stride = 1X1)
    b = BatchNormalization()(in_b)
    b = ReLU()(b)
    b = Conv2DTranspose(64, (3, 3), padding='same', strides=(1, 1))(b)
    b = BatchNormalization()(b)
    b = ReLU()(b)
    out_b = Conv2DTranspose(64, (3, 3), padding='same', strides=(1, 1))(b)
    return out_b


def block_B(in_b):
    # block B with con (stride = 2X2)
    b = BatchNormalization()(in_b)
    b = ReLU()(b)
    b = Conv2D(64, (3, 3), padding='same', strides=(1, 1))(b)
    b = BatchNormalization()(b)
    b = ReLU()(b)
    out_b = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(b)
    return out_b


def inverse_block_B(in_b):
    # block B with con (stride = 2X2)
    b = BatchNormalization()(in_b)
    b = ReLU()(b)
    b = Conv2DTranspose(64, (3, 3), padding='same', strides=(1, 1))(b)
    b = BatchNormalization()(b)
    b = ReLU()(b)
    out_b = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2))(b)
    return out_b


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# select project
neptune.init(project_qualified_name='amitshakarchy/muscle-formation',
             api_token=api_token)
# create experiment
with neptune.create_experiment(name='my_vae - Dyad archi',
                               description="learning rate scheduler",
                               tags=['classification', 'tf_2'],
                               upload_source_files=['../dayed_vae.py'],
                               params=PARAMS) as npt_exp:
    input_img = Input(shape=(image_size, image_size, 1), )
    print(input_img)
    x = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(input_img)
    x = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(x)
    out0 = MaxPool2D((2, 2))(x)

    # block A with conv (stride = 1X1)
    out1 = block_A(out0)
    add1 = add([out0, out1])

    # block B with con (stride = 2X2)
    out2 = block_B(add1)

    # conv operation to out1
    out1_conv = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(out1)
    add2 = add([out2, out1_conv])
    # ------------------------------------

    # block A with conv (stride = 1X1)
    out3 = block_A(add2)
    add3 = add([out3, add2])  # TODO: check this thing

    # block B with con (stride = 2X2)
    out4 = block_B(add3)

    # conv operation to out3
    out3_conv = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(out3)
    add4 = add([out4, out3_conv])
    # ------------------------------------

    # block A with conv (stride = 1X1)
    out5 = block_A(add4)
    add5 = add([out5, add4])

    # block B with con (stride = 2X2)
    out6 = block_B(add5)
    # conv operation to out3
    out5_conv = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(out5)
    add6 = add([out6, out5_conv])

    # ------------------------------------
    shape = K.int_shape(add6)
    print(shape)
    x = Flatten()(add6)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(input_img, [z_mean, z_log_var, z], name='origin_encoder')
    # encoder.summary()

    # decoder
    latent_inputs = Input(shape=(latent_dim,))
    y = Dense(np.prod(shape[1:]), name='encoder_input')(latent_inputs)
    y = Reshape((shape[1], shape[2], shape[3]))(y)

    # block B with DeConv (Stride = 2X2)
    _out6 = inverse_block_B(y)
    # conv operation to out3
    out_x_conv = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2))(y)
    _add6 = add([_out6, out_x_conv])

    # block A with DeConv (Stride = 1X1)
    _out5 = inverse_block_A(_add6)
    _add5 = add([_out5, _add6])
    # ------------------------------------
    # block B with DeConv (Stride = 2X2)
    _out4 = inverse_block_B(_add5)
    # conv operation to out3
    out5_conv = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2))(_out5)
    _add4 = add([_out4, out5_conv])

    # block A with DeConv (Stride = 1X1)
    _out3 = inverse_block_A(_add4)
    _add3 = add([_out3, _add4])
    # ------------------------------------
    # block B with DeConv (Stride = 2X2)
    _out2 = inverse_block_B(_add3)
    # conv operation to out3
    out3_conv = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2))(_out3)
    _add2 = add([_out2, out3_conv])

    # block A with DeConv (Stride = 1X1)
    _out1 = inverse_block_A(_add2)
    _add1 = add([_out1, _add2])

    out0 = Conv2DTranspose(64, (3, 3), padding='same', strides=(2, 2))(_add1)
    out0 = UpSampling2D()(out0)
    reconstruction = Conv2DTranspose(1, (3, 3), padding='same', strides=(2, 2))(out0)

    decoder = Model(latent_inputs, reconstruction, name='origin_decoder')

    encoder.summary()
    decoder.summary()

    # log model summary
    encoder.summary(print_fn=lambda x: neptune.log_text('encoder_summary', x))
    decoder.summary(print_fn=lambda x: neptune.log_text('decoder_summary', x))

    outputs = decoder(encoder(input_img))
    vae = Model(input_img, outputs, name='vae')

    reconst_loss = binary_crossentropy(K.flatten(input_img), K.flatten(outputs))
    reconst_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconst_loss + kl_loss)

    vae.add_loss(vae_loss)
    opt = PARAMS['optimizer'](PARAMS['learning_rate'])
    vae.compile(optimizer=opt)
    # vae.summary()

    callback = NeptuneCallback(neptune_experiment=npt_exp, n_batch=steps, images=x_test,
                               img_size=image_size)

    callbacks = [callback,
                 LambdaCallback(on_epoch_end=lambda epoch, logs: log_data(logs)),
                 EarlyStopping(patience=PARAMS['early_stopping'],
                               monitor='loss',
                               restore_best_weights=True),
                 ]
    if PARAMS['scheduler']:
        callbacks.append(LearningRateScheduler(lr_scheduler))

    vae.fit(
        generate_Data(directory=directory,
                      batch_size=batch_size,
                      img_size=image_size),
        validation_data=val_generator(directory=directory,
                                      batch_size=batch_size,
                                      img_size=image_size,
                                      validation_split=0.3),
        epochs=PARAMS['n_epochs'],
        shuffle=PARAMS['shuffle'],
        steps_per_epoch=steps,
        validation_steps=steps,
        callbacks=callbacks
    )
