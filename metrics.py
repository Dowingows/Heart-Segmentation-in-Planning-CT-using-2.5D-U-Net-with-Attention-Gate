
import numpy as np
def single_dice_coef(y_pred, y_true):
    # shape of y_true and y_pred: (height, width)
    intersection = np.sum(y_true * y_pred)
    if (np.sum(y_true) == 0) and (np.sum(y_pred) == 0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred))


def mean_dice_coef(y_pred, y_true):
    # shape of y_true and y_pred: (n_samples, height, width)
    batch_size = y_true.shape[0]
    mean_dice_channel = 0.
    for i in range(batch_size):
        channel_dice = single_dice_coef(y_pred[i, :, :], y_true[i, :, :])
        mean_dice_channel += channel_dice/(batch_size)
    return mean_dice_channel

def mean_dice_coef_remove_empty(y_pred, y_true):
    # shape of y_true and y_pred: (n_samples, height, width)
    batch_size = y_true.shape[0]
    mean_dice_channel = 0.
    num_no_empty = batch_size
    for i in range(batch_size):
        if (np.sum(y_true[i, :, :]) == 0):
            num_no_empty -= 1
            continue

        channel_dice = single_dice_coef(y_pred[i, :, :], y_true[i, :, :])
        mean_dice_channel += channel_dice
    
    if num_no_empty == 0:
        return None

    return mean_dice_channel/(num_no_empty)    