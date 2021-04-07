import tensorflow as tf

# define parameters
PARAMS = {'batch_size': 32,
          'n_epochs': 50,
          'shuffle': True,
          'activation': 'relu',
          'dense_units': 500,
          'dropout': 0.2,  # 0.2
          'learning_rate': 0.0001,
          'ae_learning_rate': 5e-5,
          'dc_learning_rate': 5e-4,
          'gen_learning_rate': 5e-5,
          'early_stopping': 10,
          'optimizer': tf.optimizers.Adam,
          'validation_split': 0.2,
          'img_size': 64,
          'z_dim': 100,
          'scheduler': True,
          }

api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYjQxZTExMGEtNGQzMi00MmY2LWEwYmQtMjJhMTBiYjNmOThkIn0='
