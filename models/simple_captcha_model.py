import keras
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from base.base_model import BaseModel
from config import config

class simple_captcha_model(BaseModel):
    def __init__(self, config):
        super(captcha_model, self).__init__(config)
        self.build_model()

    def build_model(self):#, loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy']):

        if os.path.isfile(self.model_path):
            self.model = load_model(self.model_path)
        else:
            print('Creating CNN model...')
            tensor_in = Input((75, 100, 3))
            tensor_out = tensor_in
            tensor_out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
            tensor_out = Dropout(0.2)(tensor_out)
            tensor_out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(tensor_out)
            tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
            tensor_out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
            tensor_out = Dropout(0.2)(tensor_out)
            tensor_out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(tensor_out)
            tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
            tensor_out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
            tensor_out = Dropout(0.2)(tensor_out)
            tensor_out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(tensor_out)
            tensor_out = BatchNormalization()(tensor_out)
            tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
            tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
            tensor_out = Dropout(0.2)(tensor_out)
            tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
            tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
            tensor_out = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
            tensor_out = BatchNormalization()(tensor_out)
            tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)

            tensor_out = Flatten()(tensor_out)
            tensor_out = Dropout(0.5)(tensor_out)
            #useCircle = 1
            if (self.config.useCircle==0):
                digit_1 = Dense(10, name='digit1', activation='softmax')(tensor_out)

                digit_2 = keras.layers.concatenate([digit_1, Dense(10, activation='relu')(tensor_out)]) 
                digit_2 = Dense(10, name='digit2', activation='softmax')(digit_2)

                digit_3 = keras.layers.concatenate([digit_1, digit_2, Dense(10, activation='relu')(tensor_out)]) 
                digit_3 = Dense(10, name='digit3', activation='softmax')(digit_3)

                digit_4 = keras.layers.concatenate([digit_1, digit_2 , digit_3, Dense(10, activation='relu')(tensor_out)]) 
                digit_4 = Dense(10, name='digit4', activation='softmax')(digit_4)

                tensor_out = [digit_1, digit_2, digit_3, digit_4]
            else:
                circle1 = Dense(3, name='circle1', activation='softmax')(tensor_out)
                circle2 = Dense(3, name='circle2', activation='softmax')(tensor_out)
                circle3 = Dense(3, name='circle3', activation='softmax')(tensor_out)
                circle4 = Dense(3, name='circle4', activation='softmax')(tensor_out)

                digit_1 = Dense(10, name='digit1', activation='softmax')(tensor_out)

                digit_2 = keras.layers.concatenate([digit_1, Dense(10, activation='relu')(tensor_out)]) 
                digit_2 = Dense(10, name='digit2', activation='softmax')(digit_2)

                digit_3 = keras.layers.concatenate([digit_1, digit_2, Dense(10, activation='relu')(tensor_out)]) 
                digit_3 = Dense(10, name='digit3', activation='softmax')(digit_3) 

                digit_4 = keras.layers.concatenate([digit_1, digit_2 , digit_3, Dense(10, activation='relu')(tensor_out), circle1, circle2, circle3, circle4]) 
                digit_4 = Dense(10, name='digit4', activation='softmax')(digit_4)

                tensor_out = [digit_1, digit_2, digit_3, digit_4, circle1, circle2, circle3, circle4]
                #tensor_out = [circle1, circle2, circle3, circle4]


            self.model = Model(inputs=tensor_in, outputs=tensor_out)
        self.model.compile(loss=self.config.loss, optimizer=self.config.optimizer, metrics=self.config.metrics)
    
    def train(self,x_train, y_train, x_val, y_val, batch_size, epoch):
        v_epochs = epoch
        v_batch_size = batch_size
        now = time.strftime("%y%m%d_%H:%M:%S")
        c_tensorBoard = keras.callbacks.TensorBoard(log_dir='./logs_dev/'+now + '_e' + str(v_epochs) + '_b' + str(v_batch_size), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
        callbacks_list = [c_tensorBoard]
        print('Start training...')
        train_history = self.model.fit(x_train, y_train, batch_size=v_batch_size, epochs=v_epochs, verbose=1, validation_data=(x_val, y_val), callbacks=callbacks_list)
        self.model.save('./models_dev/' + now + '_e' + str(v_epochs) + '_b' + str(v_batch_size) + '.h5')

    def predict_tmp(self, di):
        ok = 0
        for data_idx, key in enumerate(di.keys()):    
            if (data_idx > 899):        
                x_pred = []
                test_img = Image.open(key).convert('RGB')
                x_pred.append(np.array(test_img)/255)
                y_val = str(di[key])
                
                pred = model.predict(np.array(x_pred))
                predVal = ""
                for predC in np.argmax(pred[:4], axis=2):
                    predVal = predVal + str(predC[0])
                
                if (predVal == y_val):
                    ok=ok+1
                else:
                    print(key + ' ' + predVal + ' ' + y_val )
                #pred = model.predict(np.array(x_pred))
                #y_val = [  ]
                #if(pred==y_val):
        #            ok=ok+1
        print(ok)