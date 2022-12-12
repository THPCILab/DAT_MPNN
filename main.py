import torch
from config import get_config
import warnings
warnings.filterwarnings('ignore')


def main ():
    # parse configuration
    config, _ = get_config()
    torch.cuda.set_device('cuda:' + config.gpu)
    
    from train import Training
    from test import Testing
    
    if not config.test:
        train = Training(config)
        train.net_training_process()
    else:
        test = Testing(config)
        test.do_testing()

if __name__ == "__main__":
    main ()