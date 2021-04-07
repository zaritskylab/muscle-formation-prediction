import neptune
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from convolutional_aae import ConvAAE
from Encoders import *
from Decoders import *
from Trainers import *
from params import PARAMS, api_token
from vae import VAE
from NeptuneMethods import lr_scheduler

# def get_optimiser(x):
#     return {
#         'Adam': Adam,
#         'Nadam': Nadam,
#         'SGD': SGD
#     }[x]


# select project
neptune.init(project_qualified_name='amitshakarchy/muscle-formation',
             api_token=api_token)

# create experiment
with neptune.create_experiment(name='classification_example',
                               tags=['classification', 'tf_2'],
                               upload_source_files=['classification-example.py', 'requirements.txt'],
                               params=PARAMS) as npt_exp:
    optimizer = PARAMS['optimizer'](PARAMS['learning_rate'])

    model, volume_size, encoder_vae, encoder_mu, encoder_log_variance = sample_res_net_v0(PARAMS['img_size'],
                                                                                          PARAMS['z_dim'])
    decoder = decoder_res_net_v0(z_size=PARAMS['z_dim'], volume_size=volume_size)

    autoencoder = VAE(encoder=encoder_vae,
                      decoder=decoder,
                      encoder_mu=encoder_mu,
                      encoder_log_variance=encoder_log_variance,
                      batch_size=PARAMS['batch_size'],
                      n_epochs=PARAMS['n_epochs'],
                      shuffle=PARAMS['shuffle'],
                      optimizer=optimizer, )

    info = "with RELU activation function, no noise"

    # autoencoder.trainer = VAETrainer(autoencoder)
    autoencoder.load_data()
    autoencoder.train(npt_exp)
    # autoencoder.plot_history("batch_size _{}_")
    autoencoder.encoder.save("encoder CAE _{}_")
    autoencoder.decoder.save("decoder CAE _{}_")
    # autoencoder.discriminator.save("discriminator CAE _{}_")
