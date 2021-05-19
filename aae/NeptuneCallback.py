from __future__ import print_function
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import neptune
import matplotlib.pyplot as plt


class NeptuneCallback(Callback):
    def __init__(self, neptune_experiment, n_batch, test_generator, img_size):
        super().__init__()
        self.exp = neptune_experiment
        self.n = n_batch
        self.current_epoch = 0
        self.X_images, self.Y_images = test_generator.next()
        self.img_size = img_size

    def on_batch_end(self, batch, logs=None):
        x = (self.current_epoch * self.n) + batch
        self.exp.send_metric(channel_name='batch end loss', x=x, y=logs['loss'])

    def on_epoch_end(self, epoch, logs=None):
        self.exp.send_metric('epoch end loss', logs['loss'])

        msg_loss = 'End of epoch {}, categorical crossentropy loss is {:.4f}'.format(epoch, logs['loss'])
        self.exp.send_text(channel_name='loss information', x=epoch, y=msg_loss)


        if self.current_epoch % 5 == 0:
            # Reconstruction
            n_imgs = 10  # how many images we will display
            x_test_decoded = self.model.predict(self.X_images[:n_imgs])
            fig = plt.figure(figsize=(20, 4))
            for i in range(n_imgs):
                # display original
                ax = plt.subplot(2, n_imgs, i + 1)
                plt.imshow(self.X_images[i].reshape(self.img_size, self.img_size))
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # display reconstruction
                ax = plt.subplot(2, n_imgs, i + 1 + n_imgs)
                plt.imshow(x_test_decoded[i])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.title("epoch #{}".format(epoch))
            neptune.log_image('predictions', fig)

            plt.close('all')

        self.current_epoch += 1
