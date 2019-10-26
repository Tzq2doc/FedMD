import sys
import argparse
from tensorflow.keras.callbacks import EarlyStopping

from data_utils import load_MNIST_data
from Neural_Networks import cnn_2layer_fc_model, cnn_3layer_fc_model


def parseArg():
    parser = argparse.ArgumentParser(description='Train an array of Neural Networks on either MNIST or CIFAR')
    parser.add_argument('-conf', metavar='conf_file', nargs=1, 
                        help='the config file for training, \
                        for training on MNIST, the default conf_file is ./conf/pretrain_MNIST.json, \
                        for training on CIFAR, the default conf_file is ./conf/pretrain_CIFAR.json.')

    conf_file = "./conf/pretrain_MNIST.json"
    
    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
    return conf_file



def train_models(models, X_train, y_train, X_test, y_test, 
                 min_delta = 0.001, patience = 3, batch_size = 128, epochs = 20, is_shuffle=True, verbose = 1, 
                 is_show = False, save_dir = "", save_names = None):
    '''
    Train an array of models on the same dataset. 
    We use early termination to speed up training. 
    '''
    
    resulting_val_acc = []
    for n, model in enumerate(models):
        print("Training model ", n)
        model.fit(X_train, y_train, 
                  validation_data = [X_test, y_test],
                  callbacks=[EarlyStopping(monitor='val_acc', min_delta=min_delta, patience=patience)],
                  batch_size = batch_size, epochs = epochs, shuffle=is_shuffle, verbose = verbose
                 )
        
        resulting_val_acc.append(model.history.history["val_acc"][-1])
        if save_names is None:
            file_name = save_dir + "model_{0}".format(n) + ".h5"
        else:
            file_name = save_dir + name[n] + ".h5"
        model.save(file_name)
    
    if is_show:
        print("pre-train accuracy: ")
        print(resulting_val_acc)

        
models = {"2_layer_CNN": cnn_2layer_fc_model, 
          "3_layer_CNN": cnn_3layer_fc_model}        
        
        
if __name__ == "__main__":
    conf_file =  parseArg()
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())
    dataset = conf_dict["data_type"]
    model_config = conf_dict["models"]
    train_params = conf_dict["train_params"]
    del conf_dict
    
    
    if dataset == "MNIST":
        input_shape = (28,28)
        X_train, y_train, X_test, y_test = load_MNIST_data(standarized = True, 
                                                           verbose = True)
    
    else:
        print("Unknown dataset. Program stopped.")
        sys.exit()
    
    pretrain_models = []
    for item in model_config:
        name = item["model_type"]
        model_params = item["params"]
        tmp = models[name](input_shape=input_shape,
                          **model_params)
        pretrain_models.append(tmp)
    
    train_models(pretrain_models, X_train, y_train, X_test, y_test, 
                 save_dir = "", save_names = None, is_show=True,
                 **train_params
                 )
