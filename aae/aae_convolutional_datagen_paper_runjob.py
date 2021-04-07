
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches
from runjob_datagen import aae_generator, aae_load_images, generate_Data
from Encoders import *
from Decoders import *
from Discriminators import *
from DirectoriesHelper import *
from Trainers import *
import numpy as np

class ConvAAE():

    def __init__(self):
        # Set random seed
        random_seed = 42
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        self.img_size = 64
        self.z_dim = 100
        self.batch_size = 16
        self.n_epochs = 51
        self.volume_size = 0

        # Define optimizers
        self.ae_learning_rate = 0.0005 #0.0005
        self.dc_learning_rate = 0.0005 #0.0005
        self.gen_learning_rate = 0.0005 #0.00005

        self.ae_optimizer = tf.keras.optimizers.Adam(self.ae_learning_rate)
        self.dc_optimizer = tf.keras.optimizers.Adam(self.dc_learning_rate)  # 4
        self.gen_optimizer = tf.keras.optimizers.Adam(self.gen_learning_rate)

        self.epoch_ae_loss_avg = tf.metrics.Mean()
        self.epoch_dc_loss_avg = tf.metrics.Mean()
        self.epoch_dc_acc_avg = tf.metrics.Mean()
        self.epoch_gen_loss_avg = tf.metrics.Mean()

        # Define loss functions
        self.ae_loss_weight = 1.
        self.gen_loss_weight = 1.
        self.dc_loss_weight = 1.

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.accuracy = tf.keras.metrics.BinaryAccuracy()

        self.build_encoder = make_encoder_model_paper
        self.build_decoder = make_decoder_model_paper
        self.build_discriminator = make_discriminator_model

        self.hist = {"ae_loss": [], "ae_acc": [], "dc_loss": [], "dc_acc": [], "gen_loss": []}

        self.trainer = None



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

    def load_data(self):
        self.train_dataset = tf.data.Dataset.from_generator(aae_generator, (tf.float32))
        self.train_dataset = self.train_dataset.batch(self.batch_size)
        self.x_test = aae_load_images(image_size=self.img_size)[:30]

    def build_models(self):
        self.encoder, volume_size = self.build_encoder(z_size=self.z_dim, img_size=self.img_size)
        self.decoder = self.build_decoder(z_size=self.z_dim, volume_size=volume_size)
        self.discriminator = self.build_discriminator(self.z_dim)


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
    i = "VAE trainer"
    info = "with RELU activation function, no noise"
    autoencoder = ConvAAE()
    autoencoder.build_encoder = sample_res_net_v0
    autoencoder.build_decoder = decoder_res_net_v0
    autoencoder.build_models()
    autoencoder.trainer = VAETrainer(autoencoder)
    autoencoder.load_data()
    autoencoder.open_dirs("outputs _{}_".format(i))
    autoencoder.write_metadata(info)
    autoencoder.trainer.train()
    autoencoder.plot_history("batch_size _{}_".format(i))
    autoencoder.encoder.save("encoder CAE _{}_".format(i))
    autoencoder.decoder.save("decoder CAE _{}_".format(i))
    autoencoder.discriminator.save("discriminator CAE _{}_".format(i))


