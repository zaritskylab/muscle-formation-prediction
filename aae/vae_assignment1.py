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

latent_dim = PARAMS['z_dim']
image_size = 64
directory = "../../data/images/train"
if not path.exists(directory):
    directory = "muscle-formation-diff/data/images/train"
batch_size = PARAMS['batch_size']
steps = len(listdir(directory)) // batch_size
x_test = aae_load_images(image_size=image_size)[:30]


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
with neptune.create_experiment(name='my_vae - assignment1 archi',
                               description="learning rate scheduler",
                               tags=['classification', 'tf_2'],
                               upload_source_files=['classification-example.py', 'requirements.txt'],
                               params=PARAMS) as npt_exp:
    input_img = Input(shape=(image_size, image_size, 1), )
    #
    # h = Conv2D(16, kernel_size=3, activation='relu', padding='same', strides=2)(input_img)
    # enc_ouput = Conv2D(32, kernel_size=3, activation='relu', padding='same', strides=2)(h)
    #
    # shape = K.int_shape(enc_ouput)
    # x = Flatten()(enc_ouput)
    # x = Dense(16, activation='relu')(x)

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

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(input_img, [z_mean, z_log_var, z], name='origin_encoder')
    encoder.summary()

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = Conv2DTranspose(32, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(16, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(16, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    dec_output = Conv2DTranspose(1, kernel_size=3, activation='relu', padding='same')(x)
    decoder = Model(latent_inputs, dec_output, name='origin_decoder')

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
    vae.compile(optimizer=opt)  # rmsprop
    vae.summary()

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
