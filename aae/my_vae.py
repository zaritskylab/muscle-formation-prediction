from os import listdir
from os import path
import neptune
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten, Lambda, Reshape, Conv2DTranspose, BatchNormalization, MaxPool2D, \
    Dropout, Dense, Input, ReLU, add, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from tensorflow.python.keras.callbacks import LambdaCallback, EarlyStopping, LearningRateScheduler
from NeptuneCallback import NeptuneCallback
from Trainers import log_data
from params import PARAMS, api_token
from NeptuneMethods import lr_scheduler

from checking_image_data_generator import get_generators

latent_dim = PARAMS['z_dim']
image_size = 64
directory = "../../data/images/train"
if not path.exists(directory):
    directory = "muscle-formation-diff/data/images/train"
batch_size = PARAMS['batch_size']
steps = len(listdir(directory)) // batch_size
n_filters = 2


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
with neptune.create_experiment(name='my_vae - resnet archi',
                               description="learning rate scheduler",
                               tags=['classification', 'tf_2'],
                               upload_source_files=['classification-example.py', 'requirements.txt'],
                               params=PARAMS) as npt_exp:
    input_img = Input(shape=(image_size, image_size, 1), )

    # <editor-fold desc="block 1">
    '''block_1'''
    b1_cnv2d_1 = Conv2D(filters=n_filters * 16, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False, name='b1_cnv2d_1', kernel_initializer='normal')(input_img)
    b1_relu_1 = ReLU(name='b1_relu_1')(b1_cnv2d_1)
    b1_bn_1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b1_bn_1')(b1_relu_1)  # size: 32*32

    b1_cnv2d_2 = Conv2D(filters=n_filters * 32, kernel_size=(1, 1), strides=(2, 2), padding='same',
                        use_bias=False, name='b1_cnv2d_2', kernel_initializer='normal')(b1_bn_1)
    b1_relu_2 = ReLU(name='b1_relu_2')(b1_cnv2d_2)
    b1_out = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b1_out')(b1_relu_2)  # size: 16*16
    # </editor-fold>

    # <editor-fold desc="block 2">
    '''block 2'''
    b2_cnv2d_1 = Conv2D(filters=n_filters * 32, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        use_bias=False, name='b2_cnv2d_1', kernel_initializer='normal')(b1_out)
    b2_relu_1 = ReLU(name='b2_relu_1')(b2_cnv2d_1)
    b2_bn_1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b2_bn_1')(b2_relu_1)  # size: 16*16

    b2_add = add([b1_out, b2_bn_1])  #

    b2_cnv2d_2 = Conv2D(filters=n_filters * 64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False, name='b2_cnv2d_2', kernel_initializer='normal')(b2_add)
    b2_relu_2 = ReLU(name='b2_relu_2')(b2_cnv2d_2)
    b2_out = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b2_bn_2')(b2_relu_2)  # size: 8*8
    # </editor-fold>

    # <editor-fold desc="block 3">
    '''block 3'''
    b3_cnv2d_1 = Conv2D(filters=n_filters * 64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        use_bias=False, name='b3_cnv2d_1', kernel_initializer='normal')(b2_out)
    b3_relu_1 = ReLU(name='b3_relu_1')(b3_cnv2d_1)
    b3_bn_1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b3_bn_1')(b3_relu_1)  # size: 8*8

    b3_add = add([b2_out, b3_bn_1])  #

    b3_cnv2d_2 = Conv2D(filters=n_filters * 128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                        use_bias=False, name='b3_cnv2d_2', kernel_initializer='normal')(b3_add)
    b3_relu_2 = ReLU(name='b3_relu_2')(b3_cnv2d_2)
    b3_out = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b3_out')(b3_relu_2)  # size: 4*4
    # </editor-fold>

    shape = tf.keras.backend.int_shape(b3_out)

    # <editor-fold desc="block 4">
    '''block 4'''
    b4_avg_p = GlobalAveragePooling2D()(b3_out)
    flattened = Flatten()(b4_avg_p)
    output = Dense(PARAMS['dense_units'], name='model_output', activation='softmax',
                   kernel_initializer='he_uniform')(flattened)  # softmax

    z_mean = Dense(latent_dim, name='z_mean')(flattened)
    z_log_var = Dense(latent_dim, name='z_log_var')(flattened)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    # </editor-fold>

    encoder = Model(input_img, [z_mean, z_log_var, z], name='origin_encoder')
    encoder.summary()

    # <editor-fold desc="decoder">
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    b4_out = Reshape((shape[1], shape[2], shape[3]))(x)

    '''block 4'''

    # #b4_avg_p = GlobalAveragePooling2D()(y)
    # b4_out = Dense(z_size, name='model_output', activation='softmax',
    #                kernel_initializer='he_uniform')(y)

    '''block 3'''
    b3_cnv2d_1 = Conv2D(filters=n_filters * 128, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        use_bias=False, name='b3_cnv2d_1', kernel_initializer='normal')(b4_out)
    b3_relu_1 = ReLU(name='b3_relu_1')(b3_cnv2d_1)
    b3_bn_1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b3_bn_1')(b3_relu_1)  # size: 4*4

    b3_add = add([b4_out, b3_bn_1])  #

    b3_cnv2d_2 = UpSampling2D((2, 2))(b3_add)
    b3_cnv2d_2 = Conv2D(filters=n_filters * 64, kernel_size=(3, 3), padding='same',
                        use_bias=False, name='b3_cnv2d_2', kernel_initializer='normal')(b3_cnv2d_2)
    b3_relu_2 = ReLU(name='b3_relu_2')(b3_cnv2d_2)
    b3_out = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b3_out')(b3_relu_2)  # size: 8*8

    '''block 2'''
    b2_cnv2d_1 = Conv2D(filters=n_filters * 64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        use_bias=False, name='b2_cnv2d_1', kernel_initializer='normal')(b3_out)
    b2_relu_1 = ReLU(name='b2_relu_1')(b2_cnv2d_1)
    # b2_bn_1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b2_bn_1')(b2_relu_1)  # size: 8*8

    b2_add = add([b3_out, b2_relu_1])  #

    b2_cnv2d_2 = UpSampling2D((2, 2))(b2_add)
    b2_cnv2d_2 = Conv2D(filters=n_filters * 32, kernel_size=(3, 3), padding='same',
                        use_bias=False, name='b2_cnv2d_2', kernel_initializer='normal')(b2_cnv2d_2)
    b2_relu_2 = ReLU(name='b2_relu_2')(b2_cnv2d_2)
    # b2_out = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b2_bn_2')(b2_relu_2)  # size: 16*16

    '''block_1'''
    b1_cnv2d_1 = UpSampling2D((2, 2))(b2_relu_2)
    b1_cnv2d_1 = Conv2D(filters=n_filters * 32, kernel_size=(3, 3), padding='same',
                        use_bias=False, name='b1_cnv2d_1', kernel_initializer='normal')(b1_cnv2d_1)
    b1_relu_1 = ReLU(name='b1_relu_1')(b1_cnv2d_1)
    # b1_bn_1 = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b1_bn_1')(b1_relu_1)  # size: 32*32

    b1_cnv2d_2 = UpSampling2D((2, 2))(b1_relu_1)
    b1_cnv2d_2 = Conv2D(filters=n_filters * 16, kernel_size=(1, 1), padding='same',
                        use_bias=False, name='b1_cnv2d_2', kernel_initializer='normal')(b1_cnv2d_2)
    b1_relu_2 = ReLU(name='b1_relu_2')(b1_cnv2d_2)
    # b1_out = BatchNormalization(epsilon=1e-3, momentum=0.999, name='b1_out')(b1_relu_2)  # size: 64*64

    reconstruction = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same')(b1_relu_2)
    # </editor-fold>

    decoder = Model(latent_inputs, reconstruction, name='origin_decoder')
    decoder.summary()

    # log model summary
    encoder.summary(print_fn=lambda x: neptune.log_text('encoder_summary', x))
    decoder.summary(print_fn=lambda x: neptune.log_text('decoder_summary', x))

    outputs = decoder(encoder(input_img))
    vae = Model(input_img, outputs, name='vae')


    def get_vae_loss(input, x_decoded_mean, encoded_sigma, encoded_mean):
        xent_loss = tf.reduce_mean(mean_squared_error(input, x_decoded_mean))
        kl_loss = 0.5 * tf.reduce_sum(tf.square(encoded_mean) + tf.square(encoded_sigma) - tf.math.log(
            tf.square(encoded_sigma)) - 1, -1)
        return xent_loss + kl_loss


    reconst_loss = mean_squared_error(K.flatten(input_img), K.flatten(outputs))
    reconst_loss *= image_size * image_size
    kl_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(tf.exp(z_log_var)) - z_log_var - 1, axis=-1)

    # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss *= -0.5
    vae_loss = K.mean(reconst_loss + kl_loss)

    vae.add_loss(get_vae_loss(input=input_img, x_decoded_mean=outputs, encoded_sigma=z_log_var, encoded_mean=z_mean))
    opt = PARAMS['optimizer'](PARAMS['learning_rate'])
    vae.compile(optimizer=opt)  # rmsprop
    vae.summary()

    train_generator, validation_generator, test_generator = get_generators()

    callback = NeptuneCallback(neptune_experiment=npt_exp, n_batch=steps, test_generator=test_generator,
                               img_size=image_size)
    callbacks = [callback,
                 LambdaCallback(on_epoch_end=lambda epoch, logs: log_data(logs)),
                 EarlyStopping(patience=PARAMS['early_stopping'],
                               monitor='loss',
                               restore_best_weights=True),
                 ]
    if PARAMS['scheduler']:
        callbacks.append(LearningRateScheduler(lr_scheduler))

    vae.fit(train_generator, steps_per_epoch=len(train_generator),
            validation_data=validation_generator, validation_steps=len(validation_generator),
            epochs=PARAMS['n_epochs'],
            shuffle=PARAMS['shuffle'],
            callbacks=callbacks)

    vae.save("my_vae_exp_272")
