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
    test_data_dir = r'D:\GAN\mine\pix2pix-label_map-to-image-translation\dataset\dot_char_crop_simple'
    history_dir = r'history\dot_char_crop_simple\202307041116_epoch500numD2_inputc3_outputc3'
    

    #config_dir = os.path.join(history_dir, 'config.txt')
    #settings = Settings(config_dir)

    ckpt_dir = os.path.join(history_dir, 'ckpt', 'epoch_480.pt')
    ckpt = torch.load(ckpt_dir)

    settings = ckpt['config']

    settings.debug_mode = True

    epoch = settings.epoch
    num_D = settings.num_D
    n_layer_D = settings.n_layer_D
    input_nc = settings.input_nc
    input_h = settings.input_h
    input_w = settings.input_w
    label_nc = settings.label_nc
    output_nc = settings.output_nc
    n_downsampling = settings.n_downsampling

    # test dataloader
    test_dataset = MyDataset(
        data_path=test_data_dir,
        phase='test',
        input_nc=input_nc, 
        target_size=(input_w, input_h)
    )
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
        save_tensor_2(fake_image.detach(), save_dir, filename, epoch)

        #pred_real = netD.forward(fake_image.detach())
        #save_list_feature_maps(pred_real[0],os.path.join(settings.project_dir, 'visualization', 'D_features'), filename, 0)
