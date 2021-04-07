from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.callbacks import LambdaCallback, EarlyStopping
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from os import path, listdir

from Scripts.aae.NeptuneCallback import NeptuneCallback
from Scripts.aae.Trainers import log_data
from Scripts.aae.convolutional_aae import AE
from Scripts.aae.datagen import generate_Data, val_generator
from Scripts.aae.params import PARAMS


class VAE(AE):
    def __init__(self, encoder, decoder, encoder_mu,
                 encoder_log_variance, learning_rate=0.001,
                 img_size=64, z_dim=100,
                 batch_size=16, n_epochs=51,
                 shuffle=True,
                 loss=binary_crossentropy, optimizer=Adam):
        super(VAE, self).__init__(encoder, decoder, img_size=img_size, z_dim=z_dim, batch_size=batch_size,
                                  n_epochs=n_epochs,
                                  shuffle=shuffle)
        self.optimizer = optimizer(learning_rate)
        self.loss = loss
        self.mu = encoder_mu
        self.log_variance = encoder_log_variance

        # Instantiate VAE
        inputs = Input(shape=(self.img_size, self.img_size, 1))
        vae_outputs = self.decoder(self.encoder(inputs))
        self.vae = Model(inputs, vae_outputs, name='vae')
        self.vae.summary()

        reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(inputs), K.flatten(vae_outputs))
        reconstruction_loss *= self.img_size ** 2
        kl_loss = 1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        self.vae_loss = K.mean(reconstruction_loss + kl_loss)

        # Define loss

    def kl_reconstruction_loss(self, true, pred):
        # Reconstruction loss
        reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * 64 ** 2
        # KL divergence loss
        kl_loss = 1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # Total loss = 50% rec + 50% KL divergence loss
        return K.mean(reconstruction_loss + kl_loss)

    def vae_loss_function(self, x, x_recon, kl_weight=0.0005):
        latent_loss = 0.5 * tf.reduce_sum(tf.exp(self.log_variance) + tf.square(self.mu) - 1.0 - self.log_variance,
                                          axis=1)
        latent_loss = tf.reduce_mean(latent_loss)
        reconstruction_loss = tf.reduce_mean(tf.abs(x - x_recon), axis=-1)
        vae_loss = kl_weight * latent_loss + reconstruction_loss
        return vae_loss

    def train(self, npt_exp):
        directory = "muscle-formation-diff/data/images/train"
        if not path.exists(directory):
            directory = "../../data/images/train"
        steps = len(listdir(directory)) // self.batch_size

        callbacks = NeptuneCallback(neptune_experiment=npt_exp, n_batch=steps, images=self.x_test[:20],
                                    img_size=self.img_size)

        tf.config.experimental_run_functions_eagerly(True)
        # self.vae.add_loss(self.kl_reconstruction_loss)
        self.vae.compile(optimizer='adam', loss=self.vae_loss_function, metrics=['accuracy'])

        # self.vae.compile(optimizer=self.optimizer, loss=self.loss_func)
        history = self.vae.fit(
            generate_Data(directory=directory,
                          batch_size=self.batch_size,
                          img_size=self.img_size),
            validation_data=val_generator(directory=directory,
                                          batch_size=self.batch_size,
                                          img_size=self.img_size,
                                          validation_split=0.3),
            epochs=self.n_epochs,
            steps_per_epoch=steps,
            validation_steps=steps,
            callbacks=[callbacks,
                       LambdaCallback(on_epoch_end=lambda epoch, logs: log_data(logs)),
                       EarlyStopping(patience=PARAMS['early_stopping'],
                                     monitor='loss',
                                     restore_best_weights=True)]
            # [LambdaCallback(on_epoch_end=lambda epoch, logs: log_data(logs)),
            #            EarlyStopping(patience=PARAMS['early_stopping'],
            #                                          monitor='accuracy',
            #                                          restore_best_weights=True)]
            # LearningRateScheduler(lr_scheduler)]
        )
        return self.vae, history
