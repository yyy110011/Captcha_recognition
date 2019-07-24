from dotmap import DotMap
import time

def config():

    m = DotMap()
    m.useCircle = 1
    m.data_path = 'label.txt'

    m.trainer.batch_size = 128
    m.trainer.num_epochs = 200
    m.trainer.validation_split = .2
    m.trainer.verbose_training = 1


    m.model.loss = 'categorical_crossentropy'
    m.model.optimizer = 'Adamax'
    m.model.metrics = ['accuracy']

    now = time.strftime("%y%m%d_%H:%M:%S")
    m.callbacks.tensorboard_log_dir = './logs/'+now + '_e' + str(m.trainer.num_epochs) + '_b' + str(m.trainer.batch_size)


