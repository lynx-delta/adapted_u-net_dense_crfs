
from keras import layers, Input
from keras.models import Model


def u_net_model(img_height, img_width, input_chn, n_classes, act_func='elu',
                regularizer='dropout', dropoutrate=0.1):
    '''
    U-Net (encoder-decoder) fully convolutional network
    img_height: image height in pixels ==> height/32 must be an integer
    img_width: image width in pixels ==> width/32 must be an integer
    img_depth: number of input channels
    n_classes: number of classes in ground truth image
    act_func: activation function (layers)
              default = 'elu'
    regularizer: batch normalisation (batchnorm) or dropout (dropout)
                 default = 'dropout'
    dropoutrate: dropout rate (1 means 100%)
                 default = 0.1
    '''
    
    w_init = 'glorot_normal'
    kernel_size = (3, 3)
    
    # Downsampling 1
    netw_input = Input(shape=(img_height, img_width, input_chn))
    conv_d_1 = layers.BatchNormalization(axis=-1)(netw_input)
    conv_d_1 = layers.Conv2D(64, kernel_size, strides=(1, 1), padding='same',
                             kernel_initializer=w_init)(conv_d_1)
    if regularizer == 'batchnorm':
        conv_d_1 = layers.BatchNormalization(axis=-1)(conv_d_1)
    conv_d_1 = layers.Activation(act_func)(conv_d_1)
    conv_d_1 = layers.Conv2D(64, kernel_size, strides=(1, 1), padding='same',
                             kernel_initializer=w_init)(conv_d_1)
    if regularizer == 'batchnorm':
        conv_d_1 = layers.BatchNormalization(axis=-1)(conv_d_1)
    conv_d_1 = layers.Activation(act_func)(conv_d_1)
    pool_1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_d_1)
    if regularizer == 'dropout':
        pool_1 = layers.Dropout(dropoutrate)(pool_1)
    
    # Downsampling 2
    conv_d_2 = layers.Conv2D(128, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(pool_1)
    if regularizer == 'batchnorm':
        conv_d_2 = layers.BatchNormalization(axis=-1)(conv_d_2)
    conv_d_2 = layers.Activation(act_func)(conv_d_2)
    conv_d_2 = layers.Conv2D(128, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_d_2)
    if regularizer == 'batchnorm':
        conv_d_2 = layers.BatchNormalization(axis=-1)(conv_d_2)
    conv_d_2 = layers.Activation(act_func)(conv_d_2)
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_d_2)
    if regularizer == 'dropout':
        pool_2 = layers.Dropout(dropoutrate)(pool_2)
    
    # Downsampling 3
    conv_d_3 = layers.Conv2D(256, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(pool_2)
    if regularizer == 'batchnorm':
        conv_d_3 = layers.BatchNormalization(axis=-1)(conv_d_3)
    conv_d_3 = layers.Activation(act_func)(conv_d_3)
    conv_d_3 = layers.Conv2D(256, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_d_3)
    if regularizer == 'batchnorm':
        conv_d_3 = layers.BatchNormalization(axis=-1)(conv_d_3)
    conv_d_3 = layers.Activation(act_func)(conv_d_3)
    pool_3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_d_3)
    if regularizer == 'dropout':
        pool_3 = layers.Dropout(dropoutrate)(pool_3)
    
    # Downsampling 4
    conv_d_4 = layers.Conv2D(512, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(pool_3)
    if regularizer == 'batchnorm':
        conv_d_4 = layers.BatchNormalization(axis=-1)(conv_d_4)
    conv_d_4 = layers.Activation(act_func)(conv_d_4)
    conv_d_4 = layers.Conv2D(512, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_d_4)
    if regularizer == 'batchnorm':
        conv_d_4 = layers.BatchNormalization(axis=-1)(conv_d_4)
    conv_d_4 = layers.Activation(act_func)(conv_d_4)
    pool_4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_d_4)
    if regularizer == 'dropout':
        pool_4 = layers.Dropout(dropoutrate)(pool_4)
    
    # Bottom block
    conv_b = layers.Conv2D(1024, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(pool_4)
    if regularizer == 'batchnorm':
        conv_b = layers.BatchNormalization(axis=-1)(conv_b)
    conv_b = layers.Activation(act_func)(conv_b)
    conv_b = layers.Conv2D(1024, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_b)
    if regularizer == 'batchnorm':
        conv_b = layers.BatchNormalization(axis=-1)(conv_b)
    conv_b = layers.Activation(act_func)(conv_b)
    
    # Upsampling 1
    up_1 = layers.UpSampling2D(size=(2, 2))(conv_b)
    if regularizer == 'dropout':
        up_1 = layers.Dropout(dropoutrate)(up_1)
    concat_u_1 = layers.concatenate([up_1, conv_d_4], axis=-1)
    conv_u_1 = layers.Conv2D(512, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(concat_u_1)
    if regularizer == 'batchnorm':
        conv_u_1 = layers.BatchNormalization(axis=-1)(conv_u_1)
    conv_u_1 = layers.Activation(act_func)(conv_u_1)
    conv_u_1 = layers.Conv2D(512, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_u_1)
    if regularizer == 'batchnorm':
        conv_u_1 = layers.BatchNormalization(axis=-1)(conv_u_1)
    conv_u_1 = layers.Activation(act_func)(conv_u_1)
    
    # Upsampling 2
    up_2 = layers.UpSampling2D(size=(2, 2))(conv_u_1)
    if regularizer == 'dropout':
        up_2 = layers.Dropout(dropoutrate)(up_2)
    concat_u_2 = layers.concatenate([up_2, conv_d_3], axis=-1)
    conv_u_2 = layers.Conv2D(256, kernel_size, strides=(1, 1), padding='same', 
                            kernel_initializer=w_init)(concat_u_2)
    if regularizer == 'batchnorm':
        conv_u_2 = layers.BatchNormalization(axis=-1)(conv_u_2)
    conv_u_2 = layers.Activation(act_func)(conv_u_2)
    conv_u_2 = layers.Conv2D(256, kernel_size, strides=(1, 1), padding='same', 
                            kernel_initializer=w_init)(conv_u_2)
    if regularizer == 'batchnorm':
        conv_u_2 = layers.BatchNormalization(axis=-1)(conv_u_2)
    conv_u_2 = layers.Activation(act_func)(conv_u_2)
    
    # Upsampling 3
    up_3 = layers.UpSampling2D(size=(2, 2))(conv_u_2)
    if regularizer == 'dropout':
        up_3 = layers.Dropout(dropoutrate)(up_3)
    concat_u_3 = layers.concatenate([up_3, conv_d_2], axis=-1)
    conv_u_3 = layers.Conv2D(128, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(concat_u_3)
    if regularizer == 'batchnorm':
        conv_u_3 = layers.BatchNormalization(axis=-1)(conv_u_3)
    conv_u_3 = layers.Activation(act_func)(conv_u_3)
    conv_u_3 = layers.Conv2D(128, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_u_3)
    if regularizer == 'batchnorm':
        conv_u_3 = layers.BatchNormalization(axis=-1)(conv_u_3)
    conv_u_3 = layers.Activation(act_func)(conv_u_3)
    
    # Upsampling 4
    up_4 = layers.UpSampling2D(size=(2, 2))(conv_u_3)
    if regularizer == 'dropout':
        up_4 = layers.Dropout(dropoutrate)(up_4)
    concat_u_4 = layers.concatenate([up_4, conv_d_1], axis=-1)
    conv_u_4 = layers.Conv2D(64, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(concat_u_4)
    if regularizer == 'batchnorm':
        conv_u_4 = layers.BatchNormalization(axis=-1)(conv_u_4)
    conv_u_4 = layers.Activation(act_func)(conv_u_4)
    conv_u_4 = layers.Conv2D(64, kernel_size, strides=(1, 1), padding='same',
                            kernel_initializer=w_init)(conv_u_4)
    if regularizer == 'batchnorm':
        conv_u_4 = layers.BatchNormalization(axis=-1)(conv_u_4)
    conv_u_4 = layers.Activation(act_func)(conv_u_4)
    
    # Output layer
    conv_u_out = layers.Conv2D(n_classes, (1, 1), strides=(1, 1), 
                               padding='same',
                               kernel_initializer=w_init)(conv_u_4)
    netw_output = layers.Activation('softmax')(conv_u_out)
    
    # Model
    model = Model(inputs=netw_input, outputs=netw_output)
    
    return model



if __name__ == "__main__":
    import sys
    u_net_model(*sys.argv[1:])
    
    
