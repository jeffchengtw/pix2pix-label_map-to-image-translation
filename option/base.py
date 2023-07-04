import argparse
import os
from datetime import datetime
import configparser

class ArgumentParserWrapper:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Settings for the program')
        self.parser.add_argument('--debug_mode', type=bool, default=False, help='debug mode')
        self.parser.add_argument('--dataset', type=str, default='dot_char_crop_simple', help='Name of dataset')
        self.parser.add_argument('--epoch', type=int, default=500, help='Number of epochs')
        self.parser.add_argument('--input_h', type=int, default=128, help='input height')
        self.parser.add_argument('--input_w', type=int, default=128, help='input width')
        self.parser.add_argument('--input_nc', type=int, default=3, help='Number of input channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='Number of output channels')
        self.parser.add_argument('--num_D', type=int, default=2, help='Number of discriminators')
        self.parser.add_argument('--n_layer_D', type=int, default=3, help='Number of layers in discriminators')
        self.parser.add_argument('--n_downsampling', type=int, default=1, help='Number of downsampling')

        # train parameter
        self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
        self.parser.add_argument('--decay_rate', type=float, default=0.2, help='# of iter to linearly decay learning rate to zero')
        

    def parse_args(self):
        args = self.parser.parse_args()

        # 創建目錄路徑
        project_dir = os.path.join(
            'history', 
            args.dataset, datetime.now().strftime("%Y%m%d%H%M") + 
            f'_epoch{args.epoch}numD{args.num_D}_inputc{args.input_nc}_outputc{args.output_nc}'
        )
        args.project_dir = project_dir
        os.makedirs(args.project_dir, exist_ok=True)
        # 將設定存入 config.txt
        config = configparser.ConfigParser()
        config['DEFAULT'] = vars(args)

        with open(os.path.join(project_dir, 'config.txt'), 'w') as configfile:
            config.write(configfile)

        return args
