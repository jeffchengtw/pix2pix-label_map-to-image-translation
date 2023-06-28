import os
import torch
from models.generator import Generator, GlobalGenerator
from models.discriminator import MultiscaleDiscriminator
import configparser

def parse_config_file(file_path):
    config = configparser.ConfigParser()
    # 添加一個預設的 section
    config.read_string('[DEFAULT]\n' + open(file_path, 'r').read())

    settings = {}
    for option in config['DEFAULT']:
        value = config.get('DEFAULT', option)

        # 處理數值轉換
        if option == 'datetime':
            value = str(value)
        elif value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        else:
            value = int(value)

        settings[option] = value

    return settings

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    history_dir = r'history\dot_char_crop_simple\202306281118_epoch100numD2_inputc1_outputc3'
    ckpt_dir = os.path.join(history_dir, 'ckpt', 'epoch_140.pt')
    config_dir = r'history\dot_char_crop_simple\202306281118_epoch100numD2_inputc1_outputc3\config.txt'
    settings = parse_config_file(config_dir)

    # 使用解析後的設定
    epoch = settings['epoch']
    num_D = settings['num_d']
    n_layer_D = settings['n_layer_d']
    input_nc = settings['input_nc']
    output_nc = settings['output_nc']
    inputH = settings['inputh']
    inputW = settings['inputw']
    datetime = settings['datetime']

    netG = Generator(input_nc=input_nc, output_nc=output_nc).to(device)
    netD = MultiscaleDiscriminator(input_nc=(input_nc+output_nc), num_D=num_D, n_layers=n_layer_D).to(device)

    ckpt = torch.load(ckpt_dir)
    netG.load_state_dict(ckpt['generator'])
    netD.load_state_dict(ckpt['discriminator'])
    epoch = ckpt['epoch']

    print('load model success!')
    print(f'model from epoch {epoch}')
    print('start inference') 
      