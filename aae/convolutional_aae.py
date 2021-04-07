import matplotlib.pyplot as plt
import neptune
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.metrics import Mean, BinaryAccuracy
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from datagen import *
from Encoders import *
from Decoders import *
from Discriminators import *
from DirectoriesHelper import *
from Trainers import *
import numpy as np


class AE(tf.keras.Model):
    def __init__(self, encoder, decoder, img_size=64, z_dim=100, batch_size=16, n_epochs=51, shuffle=True):
        super(AE, self).__init__()
        # Set random seed
        random_seed = 42
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        self.encoder = encoder
        self.decoder = decoder
        self.img_size = img_size
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.volume_size = 0
        self.shuffle = shuffle

        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.mse = MeanSquaredError()
        self.accuracy = BinaryAccuracy()

        self.hist = {"ae_loss": [], "ae_acc": [], "dc_loss": [], "dc_acc": [], "gen_loss": []}

        self.trainer = None

    def load_data(self, ):
        self.train_dataset = tf.data.Dataset.from_generator(aae_generator, (tf.float32))
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        self.x_test = aae_load_images(image_size=self.img_size)[:30]

    def build_models(self, build_encoder, build_decoder):
        self.encoder, volume_size = build_encoder(z_size=self.z_dim, img_size=self.img_size)
        self.decoder = build_decoder(z_size=self.z_dim, volume_size=volume_size)

        # log model summary
        self.encoder.summary(print_fn=lambda x: neptune.log_text('encoder_summary', x))
        self.decoder.summary(print_fn=lambda x: neptune.log_text('decoder_summary', x))


class ConvAAE(AE):

    def __init__(self, encoder, decoder, img_size=64, z_dim=100, batch_size=16, n_epochs=51, shuffle=True,
                 ae_learning_rate=5e-5, dc_learning_rate=5e-4, gen_learning_rate=5e-5, optimizer=Adam):
        super(ConvAAE, self).__init__(encoder, decoder, img_size=img_size, z_dim=z_dim, batch_size=batch_size,
                                      n_epochs=n_epochs,
                                      shuffle=shuffle)
        # Define optimizers
        self.ae_learning_rate = ae_learning_rate
        self.dc_learning_rate = dc_learning_rate
        self.gen_learning_rate = gen_learning_rate

        self.ae_optimizer = optimizer(self.ae_learning_rate)
        self.dc_optimizer = optimizer(self.dc_learning_rate)
        self.gen_optimizer = optimizer(self.gen_learning_rate)

        self.epoch_ae_loss_avg = Mean()
        self.epoch_dc_loss_avg = Mean()
        self.epoch_dc_acc_avg = Mean()
        self.epoch_gen_loss_avg = Mean()

        # Define loss functions
        self.ae_loss_weight = 1.
        self.gen_loss_weight = 1.
        self.dc_loss_weight = 1.

        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.mse = MeanSquaredError()
        self.accuracy = BinaryAccuracy()

    def __str__(self):
        return "image size: {}\n latent dimention size: {}\n" \
               "batch size: {}\n" \
               "n_epochs: {}\n" \
               "ae_optimizer learning rate: {}\n" \
               "dc_optimizer learning rate: {}\n" \
               "gen_optimizer learning rate: {}\n" \
               "ae_loss_weight: {}\n" \
               "gen_loss_weight: {}\n" \
               "dc_loss_weight: {}\n".format(self.img_size, self.z_dim, self.batch_size, self.n_epochs,
                                             self.ae_learning_rate, self.dc_learning_rate, self.gen_learning_rate,
                                             self.ae_loss_weight, self.dc_loss_weight, self.gen_loss_weight)

    def open_dirs(self, output_dir_name):
        print("This run's output directory name: {}".format(output_dir_name))
        self.latent_space_dir, self.reconstruction_dir, self.sampling_dir, self.experiment_dir = open_dirs(
            output_dir_name)

    def write_metadata(self, info):
        file = open(self.experiment_dir / "metadata.txt", "w")
        file.write(str(self))
        file.write(str(info))
        file.close()
        file = open(self.experiment_dir / "log.csv", "w")
        file.close()

        from contextlib import redirect_stdout
        with open('metadata.txt', 'w') as f:
            with redirect_stdout(f):
                self.encoder.summary()
                self.decoder.summary()

    def build_models(self, build_encoder, build_decoder, build_discriminator=make_discriminator_model):
        self.encoder, volume_size = build_encoder(z_size=self.z_dim, img_size=self.img_size)
        self.decoder = build_decoder(z_size=self.z_dim, volume_size=volume_size)
        self.discriminator = build_discriminator(self.z_dim)

        # log model summary
        self.encoder.summary(print_fn=lambda x: neptune.log_text('encoder_summary', x))
        self.decoder.summary(print_fn=lambda x: neptune.log_text('decoder_summary', x))

    def plot_history(self, title):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.n_epochs), self.hist["ae_loss"], label="train_ae_loss")
        plt.plot(np.arange(0, self.n_epochs), self.hist["dc_loss"], label="train_dc_loss")
        plt.plot(np.arange(0, self.n_epochs), self.hist["dc_acc"], label="train dc acc")
        plt.plot(np.arange(0, self.n_epochs), self.hist["gen_loss"], label="train gen loss")
        plt.title("Training Loss and Accuracy on Dataset: {}".format(title))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(self.experiment_dir / 'loss.png')
        plt.show()


if __name__ == '__main__':
    i = "normal convAE resnet"
    info = "with RELU activation function, no noise"
    autoencoder = ConvAAE()
    autoencoder.build_models(sample_res_net_v0, decoder_res_net_v0)
    autoencoder.trainer = VAETrainer(autoencoder)
    autoencoder.load_data()
    autoencoder.open_dirs("outputs _{}_".format(i))
    autoencoder.trainer.train()
    autoencoder.encoder.save("encoder CAE _{}_".format(i))
    autoencoder.decoder.save("decoder CAE _{}_".format(i))
    autoencoder.discriminator.save("discriminator CAE _{}_".format(i))
