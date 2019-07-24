from data_loader.captcha_data_loader import CaptchaDataLoader
from models.simple_captcha_model import SimpleCaptchaModel
from trainers.simple_trainer import SimpleTrainer
from config import config

def main():

    cfg = config()
    print('Loading Data...')
    data_loader = CaptchaDataLoader(cfg)

    print('Loading Model...')
    model = SimpleCaptchaModel(cfg)

    print('Loading Trainer...')
    trainer = SimpleTrainer(model.model, data_loader.get_train_data(), cfg)

    trainer.train()


if if __name__ == "__main__":
    main()