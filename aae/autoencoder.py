from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, \
    Conv2D, MaxPooling2D, UpSampling2D, \
    Input
import matplotlib.pyplot as plt

from Scripts.aae.datagen import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Autoencoder:

    def __init__(self):
        self.img_height = 64
        self.img_weight = 64
        self.filters = 32
        self.latent_dim = 80
        self.epochs = 200
        self.inputs = Input(shape=(self.img_height, self.img_weight, 1))
        self.learning_rate = 5e-6  # 5e-5
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.batch_size = 16
        self.validation_split = 0.2
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = Model(self.inputs, outputs=self.decoder(self.encoder(self.inputs)), name="autoencoder")

    def build_encoder(self):
        x = Conv2D(self.filters * 1, (3, 3), activation="relu", padding="same")(self.inputs)
        x = MaxPooling2D((2, 2), padding="same")(x)
        # x = BatchNormalization()(x)

        x = Conv2D(self.filters * 2, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        # x = BatchNormalization()(x)

        x = Conv2D(self.filters * 4, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        # x = BatchNormalization()(x)

        # flatten the network and then construct our latent vector
        self.volume_size = tf.keras.backend.int_shape(x)
        x = Flatten()(x)
        latent = Dense(self.latent_dim)(x)

        # build the encoder model
        encoder_model = Model(inputs=self.inputs, outputs=latent, name="encoder")
        return encoder_model

    def build_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,))
        y = Dense(np.prod(self.volume_size[1:]))(latent_inputs)
        y = Reshape((self.volume_size[1], self.volume_size[2], self.volume_size[3]))(y)

        y = Conv2D(self.filters * 4, (3, 3), activation="relu", padding="same")(y)
        y = UpSampling2D((2, 2))(y)

        y = Conv2D(self.filters * 2, (3, 3), activation="relu", padding="same")(y)
        y = UpSampling2D((2, 2))(y)
        y = Conv2D(self.filters * 1, (3, 3), activation="relu", padding="same")(y)
        y = UpSampling2D((2, 2))(y)
        decoded = Conv2D(1, (3, 3), activation="relu", padding="same")(y)

        decoder_model = Model(latent_inputs, decoded, name="decoder")
        return decoder_model

    def save_autoencoder(self, name, autoencoder):
        # serialize model to JSON
        model_json = autoencoder.to_json()
        with open("models/" + name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        autoencoder.save_weights(name + ".h5")
        print("Saved model to disk")

    def train(self, training):
        steps = len(listdir("../data/images/tr/all_images_dif")) // self.batch_size

        self.autoencoder.compile(self.optimizer, loss='MeanSquaredError', metrics=['mae'])

        # history = self.autoencoder.fit(
        #     generate_Data("../data/images/tr/all_images_dif", batch_size=self.batch_size, image_size=self.img_height),
        #     epochs=self.epochs,
        #     steps_per_epoch=steps)

        history = self.autoencoder.fit(
            imageLoader("../data/images/tr/all_images_dif", listdir("../data/images/tr/all_images_dif"), batch_size=self.batch_size, img_size=self.img_height),
            epochs=self.epochs,
            steps_per_epoch=steps)

        # history = self.autoencoder.fit(training, training, epochs=self.epochs, batch_size=self.batch_size,
        #                                validation_split=self.validation_split)

        return history

    def plot_history(self, history):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), history.history["loss"], label="train_loss")
        # plt.plot(np.arange(0, self.epochs), history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), history.history["mae"], label="train_acc")
        # plt.plot(np.arange(0, self.epochs), history.history["val_mae"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig('loss.png')
        plt.show()

    def display_model_performance(self, test, decoded_imgs):
        n = 20
        plt.figure(figsize=(20, 4))

        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(test[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, i + 1 + 2 * n)
            plt.imshow(decoded_imgs[i].reshape(self.img_weight, self.img_height))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


if __name__ == '__main__':
    my_autoencoder = Autoencoder()
    training = np.load("../data/all_dif_data_test.npy")
    # load_images("../data/images/tr/train2", listdir("../data/images/tr/train2"),
    #                        image_size=my_autoencoder.img_height)
    history = my_autoencoder.train(training)
    my_autoencoder.plot_history(history)

    # new_test = load_images("../data/images/tr/train2",listdir("../data/images/tr/train2"),image_size=my_autoencoder.img_height )
    # print(new_test)
    test = np.load("../data/all_dif_data_train.npy")[:100]*255.0
    encoded_imgs = my_autoencoder.encoder.predict(test)
    decoded_imgs = my_autoencoder.decoder.predict(encoded_imgs)
    my_autoencoder.display_model_performance(test, decoded_imgs=decoded_imgs)

