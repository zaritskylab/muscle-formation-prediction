import keras
from keras.models import model_from_json
import matplotlib.pyplot as plt


def display_reconstructed_images(test, decoded_images, img_height, img_weight):
    n = 20
    plt.figure(figsize=(20, 4))

    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(test[i + 13].reshape(img_weight, img_height))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(decoded_images[i + 13].reshape(img_weight, img_height))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def load_autoencoder(autoencoder_path, autoencoder_weights_path):
    '''
    Loads the autoencoder
    :param autoencoder_path:
    :param autoencoder_weights_path:
    :return:
    '''
    # load json and create model
    json_file = open(autoencoder_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    autoencoder = model_from_json(loaded_model_json)

    # load weights into new model
    autoencoder.load_weights(autoencoder_weights_path)
    print("Loaded model from disk")
    return autoencoder


def load_model(path):
    return keras.models.load_model(path, compile=False)
