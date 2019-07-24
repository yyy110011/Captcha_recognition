from base.base_trainer import BaseTrainer

from base.base_train import BaseTrain
import os
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard



class SimpleCaptchaTrainer(BaseTrainer):
    def __init__(self, model, data, config):
        super(SimpleCaptchaTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()
    
    def init_callbacks(self):
        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                #write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

    def train(self):
        history = self.model.fit(
            self.data[0], self.data[1],
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            batch_size=self.config.trainer.batch_size,
            validation_split=self.config.trainer.validation_split,
            callbacks=self.callbacks,
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss_1.extend(history.history['val_digit1_loss'])
        self.val_acc_1.extend(history.history['val_digit1_acc'])

        self.val_loss_2.extend(history.history['val_digit2_loss'])
        self.val_acc_2.extend(history.history['val_digit2_acc'])

        self.val_loss_3.extend(history.history['val_digit3_loss'])
        self.val_acc_3.extend(history.history['val_digit3_acc'])

        self.val_loss_4.extend(history.history['val_digit4_loss'])
        self.val_acc_4.extend(history.history['val_digit4_acc'])