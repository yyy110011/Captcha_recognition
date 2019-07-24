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
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )
