import csv
import time
from os import listdir
from os import path
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping, LearningRateScheduler
from params import PARAMS
from datagen import *
from Encoders import *
from Decoders import *
from Discriminators import *
from DirectoriesHelper import *
from NeptuneCallback import NeptuneCallback
import neptune
import numpy as np


def log_data(logs):
    # neptune.log_metric('epoch_accuracy', logs['accuracy'])
    # neptune.log_metric('epoch_val_accuracy', logs['val_accuracy'])
    neptune.log_metric('epoch_loss', logs['loss'])
    neptune.log_metric('epoch_val_loss', logs['val_loss'])


def disp_latent_space(encoder, x_test, latent_space_dir, epoch):
    # Latent Space
    x_test_encoded = encoder(x_test, training=False)

    fig = plt.figure()
    ax = plt.subplot(111, aspect='equal')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    handles = [mpatches.Circle((0, 0))]
    ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45),
              fancybox=True, loc='center left')
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], s=2)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

    plt.savefig(latent_space_dir / ('epoch_%d.png' % epoch))
    plt.close('all')


def disp_reconstruction(encoder, decoder, x_test, reconstruction_dir, epoch, img_size):
    # Reconstruction
    n_digits = 20  # how many digits we will display
    x_test_decoded = decoder(encoder(x_test[:n_digits], training=False),
                             training=False)
    x_test_decoded = np.reshape(x_test_decoded, [-1, img_size, img_size])  # *255
    fig = plt.figure(figsize=(20, 4))
    for i in range(n_digits):
        # display original
        ax = plt.subplot(2, n_digits, i + 1)
        plt.imshow(x_test[i].reshape(img_size, img_size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n_digits, i + 1 + n_digits)
        plt.imshow(x_test_decoded[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(reconstruction_dir / ('epoch_%d.png' % epoch))

    plt.close('all')


def disp_sampling(z_dim, decoder, img_size, sampling_dir, epoch):
    # Sampling
    x_points = np.linspace(-3, 3, 20).astype(np.float32)
    y_points = np.linspace(-3, 3, 20).astype(np.float32)

    nx, ny = len(x_points), len(y_points)
    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        # z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
        z = np.random.uniform(-3, 3, (1, z_dim))
        z = np.reshape(z, (1, z_dim))
        x = decoder(z, training=False).numpy()
        ax = plt.subplot(g)
        img = np.array(x.tolist()).reshape(img_size, img_size)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    plt.savefig(sampling_dir / ('epoch_%d.png' % epoch))
    plt.close('all')


class Trainer():
    def __init__(self, autoencoder):
        self.ae = autoencoder


class AAETrainer(Trainer):
    def __init__(self, autoencoder):
        Trainer.__init__(self, autoencoder)

    def train(self):
        for epoch in range(self.ae.n_epochs):
            start = time.time()
            for batch, (batch_x) in enumerate(self.ae.train_dataset):
                ae_loss, dc_loss, dc_acc, gen_loss = self.train_step(batch_x)
                self.ae.epoch_ae_loss_avg(ae_loss)
                self.ae.epoch_dc_loss_avg(dc_loss)
                self.ae.epoch_dc_acc_avg(dc_acc)
                self.ae.epoch_gen_loss_avg(gen_loss)

            self.ae.hist["ae_loss"].append(self.ae.epoch_ae_loss_avg.result())
            self.ae.hist["dc_loss"].append(self.ae.epoch_dc_loss_avg.result())
            self.ae.hist["dc_acc"].append(self.ae.epoch_dc_acc_avg.result())
            self.ae.hist["gen_loss"].append(self.ae.epoch_gen_loss_avg.result())

            epoch_time = time.time() - start
            to_print = '{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.8f} DC_LOSS: {:.8f} DC_ACC: {:.8f} GEN_LOSS: {:.8f}' \
                .format(epoch, epoch_time,
                        epoch_time * (self.ae.n_epochs - epoch),
                        self.ae.epoch_ae_loss_avg.result(),
                        self.ae.epoch_dc_loss_avg.result(),
                        self.ae.epoch_dc_acc_avg.result(),
                        self.ae.epoch_gen_loss_avg.result())
            print(to_print)

            with open(self.ae.experiment_dir / ("log.csv"), 'a', newline='') as myfile:
                writer = csv.writer(myfile, delimiter=',')
                writer.writerow([to_print])
            # -------------------------------------------------------------------------------------------------------------
            if epoch % 10 == 0:
                latent_space = False
                if latent_space:
                    disp_latent_space(self.ae.encoder, self.ae.x_test, self.ae.latent_space_dir, epoch)

                disp_reconstruction(self.ae.encoder, self.ae.decoder, self.ae.x_test,
                                    self.ae.reconstruction_dir, epoch, self.ae.img_size)

                sample = False
                if sample:
                    disp_sampling(self.ae.z_dim, self.ae.decoder, self.ae.img_size, self.ae.sampling_dir,
                                  epoch)

    @tf.function
    def train_step(self, batch_x):
        print("step train-AAE")
        # Autoencoder
        with tf.GradientTape() as ae_tape:
            noisy = False
            if (noisy):
                noise = np.random.normal(0, 1, batch_x.shape)
                batch_x = batch_x + noise
            encoder_output = self.ae.encoder(batch_x, training=True)
            decoder_output = self.ae.decoder(encoder_output, training=True)

            # Autoencoder loss
            ae_loss = self.ae.mse(batch_x, decoder_output)

        ae_grads = ae_tape.gradient(ae_loss, self.ae.encoder.trainable_variables + self.ae.decoder.trainable_variables)
        self.ae.ae_optimizer.apply_gradients(
            zip(ae_grads, self.ae.encoder.trainable_variables + self.ae.decoder.trainable_variables))

        # -------------------------------------------------------------------------------------------------------------
        # Discriminator
        with tf.GradientTape() as dc_tape:
            # Sample random points in the latent space
            real_distribution = tf.random.normal(shape=(self.ae.batch_size, self.ae.z_dim))  # int(self.batch_size / 2)
            # Generate a half batch of new images
            encoder_output = self.ae.encoder(batch_x, training=True)  # [:int(len(batch_x) / 2)]

            # Combine the fake encodings with real encodings
            combined_encodings = tf.concat([encoder_output, real_distribution], axis=0)

            # Assemble labels discriminating real from fake image
            labels = tf.concat([tf.ones((encoder_output.shape[0], 1)), tf.zeros((real_distribution.shape[0], 1))],
                               axis=0)

            # Add random noise to the labels - important trick!
            labels += 0.05 * tf.random.uniform(labels.shape)

            # Train the discriminator
            predictions = self.ae.discriminator(combined_encodings, training=True)

            dc_loss = self.ae.cross_entropy(labels, predictions)
            dc_acc = self.ae.accuracy(labels, predictions)
            dc_grads = dc_tape.gradient(dc_loss, self.ae.discriminator.trainable_variables)
            self.ae.dc_optimizer.apply_gradients(zip(dc_grads, self.ae.discriminator.trainable_variables))

        # Generator (Encoder)
        with tf.GradientTape() as gen_tape:
            # Assemble labels that say "all real images"
            misleading_labels = tf.zeros((len(batch_x), 1))

            encoder_output = self.ae.encoder(batch_x, training=True)  # TODO: change to training = false
            predictions = self.ae.discriminator(encoder_output, training=True)
            gen_loss = self.ae.cross_entropy(misleading_labels, predictions)
        gen_grads = gen_tape.gradient(gen_loss, self.ae.encoder.trainable_variables)
        self.ae.gen_optimizer.apply_gradients(zip(gen_grads, self.ae.encoder.trainable_variables))

        return ae_loss, dc_loss, dc_acc, gen_loss


class ConvAETrainer(Trainer):
    def __init__(self, autoencoder):
        Trainer.__init__(self, autoencoder)

    def train(self):
        directory = "../../data/images/train"
        if not path.exists(directory):
            directory = "muscle-formation-diff/data/images/train"

        inputs = Input(shape=(self.ae.img_size, self.ae.img_size, 1))
        steps = len(listdir(directory)) // self.ae.batch_size

        my_ae = Model(inputs, outputs=self.ae.decoder(self.ae.encoder(inputs)), name="autoencoder")
        my_ae.compile(self.ae.ae_optimizer, loss='MeanSquaredError', metrics=['accuracy'])
        history = my_ae.fit(
            generate_Data(directory=directory,
                          batch_size=self.ae.batch_size,
                          img_size=self.ae.img_size),
            epochs=self.ae.n_epochs,
            steps_per_epoch=steps)
        disp_reconstruction(self.ae.encoder, self.ae.decoder, self.ae.x_test, self.ae.reconstruction_dir, 50)
        return my_ae, history


class VAETrainer(Trainer):
    def __init__(self, autoencoder):
        Trainer.__init__(self, autoencoder)

    @property
    def metrics(self):
        return [
            self.ae.total_loss_tracker,
            self.ae.reconstruction_loss_tracker,
            self.ae.kl_loss_tracker,
        ]

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self, model, x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    # @tf.function
    def train_step(self, model, x):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.ae.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def train(self, npt_exp):
        print("train- VAE")
        directory = "muscle-formation-diff/data/images/train"
        if not path.exists(directory):
            directory = "../../data/images/train"

        steps = len(listdir(directory)) // self.ae.batch_size

        callbacks = NeptuneCallback(neptune_experiment=npt_exp, n_batch=steps, images=self.ae.x_test[:20],
                                    img_size=self.ae.img_size)

        self.ae.vae.compile(self.ae.optimizer)  # , loss='MeanSquaredError', metrics=['accuracy']
        history = self.ae.vae.fit(
            generate_Data(directory=directory,
                          batch_size=self.ae.batch_size,
                          img_size=self.ae.img_size),
            validation_data=val_generator(directory=directory,
                                          batch_size=self.ae.batch_size,
                                          img_size=self.ae.img_size,
                                          validation_split=0.3),
            epochs=self.ae.n_epochs,
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

        # disp_reconstruction(self.ae.encoder, self.ae.decoder, self.ae.x_test, self.ae.reconstruction_dir, 50)

        return self.vae, history
