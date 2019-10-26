





def plot_history(model):
    
    """
    input : model is trained keras model.
    """
    
    fig, axes = plt.subplots(2,1, figsize = (12, 6), sharex = True)
    axes[0].plot(model.history.history["loss"], "b.-", label = "Training Loss")
    axes[0].plot(model.history.history["val_loss"], "k^-", label = "Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    
    
    axes[1].plot(model.history.history["acc"], "b.-", label = "Training Acc")
    axes[1].plot(model.history.history["val_acc"], "k^-", label = "Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    
    plt.subplots_adjust(hspace=0)
    plt.show()
    
def show_performance(model, Xtrain, ytrain, Xtest, ytest):
    y_pred = None
    print("CNN+fC Training Accuracy :")
    y_pred = model.predict(Xtrain, verbose = 0).argmax(axis = 1)
    print((y_pred == ytrain).mean())
    print("CNN+fc Test Accuracy :")
    y_pred = model.predict(Xtest, verbose = 0).argmax(axis = 1)
    print((y_pred == ytest).mean())
    print("Confusion_matrix : ")
    print(confusion_matrix(y_true = ytest, y_pred = y_pred))
    
    del y_pred