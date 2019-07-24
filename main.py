from data_loader.captcha_data_loader import CaptchaDataLoader
from models.simple_captcha_model import SimpleCaptchaModel
from trainers.simple_trainer import SimpleTrainer
from config import config

def main():

    cfg = config()
    data