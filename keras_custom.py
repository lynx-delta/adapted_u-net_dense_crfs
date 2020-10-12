
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf


def custom_categorical_crossentropy(class_weights):
    '''Custom categorical crossentropy loss function'''
    def pixelwise_loss(y_true, y_pred):
        '''Computation of weighted pixelwise loss'''
        # Initialize weights tensor
        weights = np.array(class_weights)[np.newaxis, np.newaxis, :]
        w_tensor = weights * tf.ones_like(y_true)
        # Compute loss
        epsilon = tf.constant(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        return - tf.reduce_sum(y_true * tf.log(y_pred) * w_tensor,
                               len(y_pred.get_shape()) - 1)
    return pixelwise_loss
   

def custom_softmax(input_data):
    d = K.exp(input_data - K.max(input_data, axis=-1, keepdims=True))
    return d / K.sum(d, axis=-1, keepdims=True)


def custom_categorical_accuracy(y_true, y_pred):  # currently unused
    '''Custom overall categorical accuracy'''
    # Keras original version
    return K.cast(K.equal(K.argmax(y_true, axis=-1),  
                          K.argmax(y_pred, axis=-1)),
                          K.floatx()) 


def plot_acc_loss(acc, val_acc, loss, val_loss, file_name):
    '''Function to plot training / validation accuracy and loss'''
    
    epochs = range(1, len(acc) + 1)
    
    # Overall accuracy
    plt.figure(figsize=(5, 4))
    plt.plot(epochs, acc, 'r', label='training acc')
    plt.plot(epochs, val_acc, 'b', label='validation acc')
    plt.xlabel('epochs')
    plt.ylabel('overall accuracy')
    plt.title('Training and validation accuracy')
    plt.grid()
    legend = plt.legend()
    legend.get_frame().set_alpha(1)
    plt.savefig(file_name + '_acc' + '.png', bbox_inches='tight')
    
    # Loss
    plt.figure(figsize=(5, 4))
    plt.plot(epochs, loss, 'r', label='training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()
    legend = plt.legend()
    legend.get_frame().set_alpha(1)
    plt.savefig(file_name + '_loss' + '.png', bbox_inches='tight')    



if __name__ == "__main__":
    import sys
    plot_acc_loss(*sys.argv[1:])    
