from base.base_data_loader import BaseDataLoader
import pandas as pd
import numpy as np
import PIL


class CaptchaDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(CaptchaDataLoader, self).__init__(config)
        self.X_train, self.y_train, self.X_test, self.y_test = load_data()
    
    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test


    def load_data(self):
        split_th = self.config.split_threshold
        data = pd.read_csv(self.config.data_path, header=None)
        di = dict()

        for index, row in data.iterrows():
            if (len(str(row[1])) < 4 ):
                row[1] = ('0000' + str(row[1]))[-4:]
            di[row[0]] = str(row[1])


        X_data = []
        y_train = []
        if (self.config.useCircle==0):
            yListData = [[] for _ in range(4)]
            yListVal = [[] for _ in range(4)]
            for data_idx, key in enumerate(di.keys()):
                #if data_idx > 899:
                #    break
                img = PIL.Image.open(key).convert('RGB')
                X_data.append(np.array(img)/255)
                label = []
                for index, digit in enumerate(di[key]):
                    tmp = np.zeros(10)
                    tmp[int(digit)] = 1
                    label.append(tmp)
                    if data_idx > split_th -1:
                        yListVal[index].append(tmp)

                    else:
                        yListData[index].append(tmp)
                #y_train.append(label)
        else:
            yListData = [[] for _ in range(8)]
            yListVal = [[] for _ in range(8)]
            for data_idx, key in enumerate(di.keys()):
                #if data_idx > 899:
                #    break
                img = PIL.Image.open(key).convert('RGB')
                X_data.append(np.array(img)/255)
                for index, digit in enumerate(di[key]):
                    tmp = np.zeros(10)
                    tmp[int(digit)] = 1
                    
                    #for number label            
                    if data_idx > split_th -1:
                        yListVal[index].append(tmp)
                    else:
                        yListData[index].append(tmp)
                    
                    #for circle label
                    tmp = np.zeros(3)
                    circleMap = [1,0,0,0,1,0,1,0,2,1]
                    tmp[circleMap[int(digit)]] = 1
                    #print(int(digit),'has ', circleMap[int(digit)], ' circles')
                    indexOfCircle = index+4
                    if data_idx > split_th -1:
                        yListVal[indexOfCircle].append(tmp)
                    else:
                        yListData[indexOfCircle].append(tmp)
                    
        X_data = np.array(X_data)    
        #y_train = (y_train)

        x_train = X_data[:split_th, :, :, :]
        x_val = X_data[split_th:, :, :, :]

        y_train = yListData
        y_val = yListVal

        print('Shape of x_train : ', np.shape(x_train))
        print('Shape of y_train : ', np.shape(y_train))
        print('Shape of x_val   : ', np.shape(x_val))
        print('Shape of y_val   : ', np.shape(y_val))

        return x_train, y_train, x_val, y_val