import os
import numpy as np

if __name__ == "__main__":
    
    

    #dpath = os.path.dirname(os.path.abspath(self.db_store_name))
    
    dpath = os.path.abspath("./MNIST")
    #make dir
    try:
        os.makedirs(dpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    a = np.arange(10)
    a = np.sin(a)
    
    print(a)