from keras.callbacks import Callback
import numpy as np
import config
import cv2
from train_generator import validation_generator, train_generator


def get_mae(actual, predicted):
    n_samples = predicted.shape[0]
    diff_sum = 0.00
    for i in range(n_samples):
        p = predicted[i][0]
        a = actual[i]
        d = abs(p - a)
        diff_sum += d
    return diff_sum / n_samples


class MAECallback(Callback):


    def on_train_begin(self, logs={}):
        self._data = []


    def on_epoch_end(self, batch, logs={}):
        with open(config.CROPPED_IMGS_INFO_FILE, 'r') as f:
            test_images_info = f.read().splitlines()[-config.VALIDATION_SIZE:]
        test_x = []
        test_y = []
        for info in test_images_info:
            weight = float(info.split(';')[1])
            test_y.append(weight)
            file_name = info.split(';')[0]
            file_path = '%s/%s' % (config.CROPPED_IMGS_DIR, file_name)
            print
            print
            print
            print file_path
            img = cv2.imread(file_path)
            img = np.resize(img, (config.RESNET50_DEFAULT_IMG_WIDTH, config.RESNET50_DEFAULT_IMG_WIDTH,3))
            test_x.append(img.__div__(255.00))
            
        
#        X_val = np.array(test_x)
#        y_val = np.array(test_y)
        
        val_batch = next(validation_generator)
        train_batch = next(train_generator)
        
        #print batch
    
        X_train = train_batch[0]
        y_train = train_batch[1]

        ytrain_predict = y_predict = np.asarray(self.model.predict(X_train))

        X_val= val_batch[0]
        y_val = val_batch[1]
        
        y_predict = np.asarray(self.model.predict(X_val))
        
        a=[]
        for i in range(len(ytrain_predict)):
            a.append(ytrain_predict[i][0])
        
        print a
        print get_mae(y_train, ytrain_predict)

        b=[]
        for i in range(len(y_predict)):
            b.append(y_predict[i][0])
        
        print b
        
#        fig, ax = plt.subplots()
#        ax.plot(x, test_y, label="actual value")
#        ax.plot(x, b, label="predicted value")
#        ax.legend()
#    
#        plt.show()
        
        val_mae = get_mae(y_val, y_predict)
        logs['val_mae'] = val_mae
        self._data.append({
            'val_mae': val_mae,
        })
        return


    def get_data(self):
        return self._data
