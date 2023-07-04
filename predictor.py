import os
import torch
from models.generator import Generator, GlobalGenerator
from models.discriminator import MultiscaleDiscriminator
from dataset import MyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.visualization import*
from option.util import*




if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data_dir = r'D:\GAN\mine\dataset\dot_char_crop_simple'
    history_dir = r'history\dot_char_crop_simple\202307010707_epoch500numD2_inputc1_outputc3'
    ckpt_dir = os.path.join(history_dir, 'ckpt', 'epoch_480.pt')
    config_dir = os.path.join(history_dir, 'config.txt')
    settings = Settings(config_dir)

    settings.debug_mode = True
    # 使用解析後的設定
    epoch = settings.epoch
    num_D = settings.num_d
    n_layer_D = settings.n_layer_d
    input_nc = settings.input_nc
    input_h = settings.input_h
    input_w = settings.input_w
    label_nc = settings.label_nc
    output_nc = settings.output_nc
    n_downsampling = settings.n_downsampling

    # test dataloader
    test_dataset = MyDataset(test_data_dir, 'test', (input_w, input_h))
    test_loader = DataLoader(test_dataset, batch_size=1)

    netG = Generator(
        input_nc=label_nc, 
        output_nc=output_nc, 
        n_downsampling=n_downsampling, 
        cfg=settings
    ).to(device)
    
    netD = MultiscaleDiscriminator(
        input_nc=(output_nc+label_nc),
        num_D=num_D, 
        n_layers=n_layer_D
    ).to(device)

    ckpt = torch.load(ckpt_dir)
    netG.load_state_dict(ckpt['generator'])
    netD.load_state_dict(ckpt['discriminator'])
    epoch = ckpt['epoch']

    print('load model success!')
    print(f'model from epoch {epoch}')
    print('start inference') 

    netG.eval()
    netD.eval()
    for batch in tqdm(test_loader):
        label = batch['label_tensor'].to(device)
        filename = batch['filename']

        fake_image = netG(label, 0, filename)
        save_dir = os.path.join(history_dir, 'pred')
        save_tensor_2(fake_image.detach(), save_dir,  f'fake_{filename}.bmp')

        pred_real = netD.forward(fake_image.detach())
        save_list_feature_maps(pred_real[0],os.path.join(settings.project_dir, 'visualization', 'D_features'), filename, 0)
