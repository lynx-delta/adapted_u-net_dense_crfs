
import json
import numpy as np
from keras import optimizers
import keras_custom
import netw_models


def model_train(file_train, file_val, model_name='u_net_model',
                act_func='elu', regularizer='dropout', dropoutrate=0.1,
                weighted_loss=True, class_weights=[1, 1, 1],
                batch_size=8, n_epochs=50,
                save_model=False, save_weights=True):
    '''
    Function for model training
    file_train: file that contains all training (training set) instances (.npz-file)
    file_val: file that contains all validation (dev set) instances (.npz-file)
    model_name: name of network model
    act_func: activation function (see Keras documentation for valid inputs)
    regularizer: 'dropout' or 'batchnorm'
    dropoutrate: dropoutrate (float between 0.0 and 1.0),
                 not considered when 'batchnorm' is used
    weighted_loss: True or False
    class_weights: if 'weighted_loss' is set to True, a list with the class weights 
                   must be provided, the length of the list must correspond with
                   the number of classes (ground truth), the position of a certain
                   class (weighting) must correspond with the ground truth
    batch_size: batch size to be used for training (must be smaller than 
                number of training instances)
    n_epochs: number of epochs the network is trained for (in this example,
              the training would be stopped after 50 epochs)
    save_model: True or False, saves the whole model (including training
                history etc.), may consume a lot of space on disk
    save_weights: True or False, saves the weights only
    '''
    
    # Load data files
    with np.load(file_train + '.npz') as data:
        train_X = data['data_X']
        train_Y = data['data_Y']
    with np.load(file_val + '.npz') as data:
        val_X = data['data_X']
        val_Y = data['data_Y']
    
    _, height, width, channels = train_X.shape
    n_classes = train_Y.shape[-1]
    
    # Definition of various input parameters
    model_args = (height, width, channels, n_classes)
    model_kwargs = {'act_func': act_func,
                    'regularizer': regularizer,
                    'dropoutrate': dropoutrate}
    if weighted_loss == True:
        loss_function = keras_custom.custom_categorical_crossentropy(class_weights)
    else:
        loss_function = 'categorical_crossentropy'
    
    # Build model
    model = netw_models.u_net_model(*model_args, **model_kwargs)
    
    # Compile model
    model.compile(loss=loss_function,
                  optimizer=optimizers.RMSprop(lr=1e-4, rho=0.9),
                  metrics=['acc'])
    
    # Model training
    model_fit = model.fit(train_X, train_Y, batch_size, n_epochs,
                          validation_data=(val_X, val_Y), shuffle=True)
    if save_model == True:
        model.save(model_name + '.h5')
    if save_weights == True:
        model.save_weights(model_name + '_weights.h5')
        with open(model_name + '_init.json', 'w') as file:
            json.dump(model_kwargs, file)
    
    # Plot averaged overall accuracy and loss
    keras_custom.plot_acc_loss(model_fit.history['acc'], 
                               model_fit.history['val_acc'],
                               model_fit.history['loss'],
                               model_fit.history['val_loss'],
                               model_name)
    
    del train_X, train_Y, val_X, val_Y

 
    
if __name__ == "__main__":
    import sys
    model_train(*sys.argv[1:])
    
    
    
    
