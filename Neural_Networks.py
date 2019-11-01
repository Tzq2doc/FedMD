from tensorflow.keras.models import Model, Sequential, clone_model, load_model
from tensorflow.keras.layers import Input, Dense, add, concatenate, Conv2D,Dropout,\
BatchNormalization, Flatten, MaxPooling2D, AveragePooling2D, Activation, Dropout, Reshape
import tensorflow as tf


def cnn_3layer_fc_model(n_classes,n1 = 128, n2=192, n3=256, dropout_rate = 0.2,input_shape = (28,28)):
    model_A, x = None, None
     
    x = Input(input_shape)
    if len(input_shape)==2: y = Reshape((input_shape[0], input_shape[1], 1))(x)
    
    y = Conv2D(filters = n1, kernel_size = (3,3), strides = 1, padding = "same", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size = (2,2), strides = 1, padding = "same")(y)

    y = Conv2D(filters = n2, kernel_size = (2,2), strides = 2, padding = "valid", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Conv2D(filters = n3, kernel_size = (3,3), strides = 2, padding = "valid", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    #y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(units = n_classes, activation = None, use_bias = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
    y = Activation("softmax")(y)


    model_A = Model(inputs = x, outputs = y)

    model_A.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3), 
                        loss = "sparse_categorical_crossentropy",
                        metrics = ["accuracy"])
    return model_A
  
def cnn_2layer_fc_model(n_classes,n1 = 128, n2=256, dropout_rate = 0.2,input_shape = (28,28)):
    model_A, x = None, None
    
    x = Input(input_shape)
    if len(input_shape)==2: y = Reshape((input_shape[0], input_shape[1], 1))(x)
    
    y = Conv2D(filters = n1, kernel_size = (3,3), strides = 1, padding = "same", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size = (2,2), strides = 1, padding = "same")(y)


    y = Conv2D(filters = n2, kernel_size = (3,3), strides = 2, padding = "valid", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    #y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    y = Dense(units = n_classes, activation = None, use_bias = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
    y = Activation("softmax")(y)


    model_A = Model(inputs = x, outputs = y)

    model_A.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3), 
                        loss = "sparse_categorical_crossentropy",
                        metrics = ["accuracy"])
    return model_A


def remove_last_layer(model, loss = "mean_absolute_error"):
    """
    Input: Keras model, a classification model whose last layer is a softmax activation
    Output: Keras model, the same model with the last softmax activation layer removed,
        while keeping the same parameters 
    """
    
    new_model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    new_model.set_weights(model.get_weights())
    new_model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3), 
                      loss = loss)
    
    return new_model