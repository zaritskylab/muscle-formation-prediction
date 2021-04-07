from params import PARAMS
import numpy as np
import neptune


def lr_scheduler(epoch):
    if epoch < 20:
        new_lr = PARAMS['learning_rate']
    else:
        new_lr = PARAMS['learning_rate'] * np.exp(0.05 * (20 - epoch))

    neptune.log_metric('learning_rate', new_lr)
    return new_lr
