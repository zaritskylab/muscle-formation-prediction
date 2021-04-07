from tensorflow import keras
import random

from tensorflow.python.keras import Model, Input

from Scripts.AELoader import display_reconstructed_images
from Scripts.aae.datagen import aae_load_images
from Scripts.aae.params import PARAMS

path = "all_outputs/my_vae_exp_261"
vae = keras.models.load_model(path, compile=False)
encoder = Model(vae.input, vae.get_layer("origin_encoder").output)
# encoder = keras.models.load_model("encoder_vae_exp_261", compile=False)
decoder_input = Input(shape=(PARAMS["z_dim"],))

decoder = Model(decoder_input, vae.get_layer("origin_decoder")(decoder_input))

encoder.save("encoder_vae_exp_261")
encoder.summary()
decoder.summary()

test = aae_load_images(image_size=64)
lst = random.sample(range(0, len(test) - 1), 50)
to_predict = test[lst]

encoded_images = encoder.predict(to_predict)
decoded_images = decoder.predict(encoded_images)
# decoded_images = vae.predict(to_predict)

display_reconstructed_images(to_predict, decoded_images, 64, 64)
