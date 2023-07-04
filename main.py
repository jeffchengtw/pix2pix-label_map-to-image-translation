import torch
import torch.optim as optim
from tqdm import tqdm
from dataset import MyDataset
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import MultiscaleDiscriminator
from criterion.loss import GANLoss, FeatureLoss
from utils.visualization import*
from logger.loss_logger import*
from option.base import*



def initial(args):

    input_nc = args.input_nc
    output_nc = args.output_nc
    input_w, input_h = args.input_w, args.input_h
    num_D = args.num_D
    n_layer_D = args.n_layer_D
    dataset = args.dataset
    n_downsampling = args.n_downsampling
    
    train_dataset = MyDataset(
        data_path=os.path.join('dataset', dataset), 
        phase='train', 
        target_size=(input_w, input_h)
    )
    label_nc = train_dataset.label_nc
    train_loader = DataLoader(train_dataset, batch_size=8)
    

    netG = Generator(
        input_nc=label_nc, 
        output_nc=output_nc, 
        n_downsampling=n_downsampling, 
        cfg=args
    ).to(device)
    
    netD = MultiscaleDiscriminator(
        input_nc=(output_nc+label_nc),
        num_D=num_D, 
        n_layers=n_layer_D
    ).to(device)

    criterionGAN = GANLoss(use_lsgan=True).to(device)
    criterionFeat = FeatureLoss(num_D=num_D, n_layers_D=n_layer_D)

    optimizer_G = optim.Adam(netG.parameters(), lr=0.001, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=0.001, betas=(0.9, 0.999))

    logger = LossLogger(log_dir='logs')

    return {
        'num_D': num_D,
        'n_layer_D': n_layer_D,
        'train_loader': train_loader,
        'netG': netG,
        'netD': netD,
        'criterionGAN': criterionGAN,
        'criterionFeat': criterionFeat,
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'logger': logger
    }

def train(training_item, epoch):
    train_loader = training_item['train_loader']
    optimizer_G = training_item['optimizer_G']
    optimizer_D = training_item['optimizer_D']
    netG = training_item['netG']
    netD = training_item['netD']
    criterionGAN = training_item['criterionGAN']
    criterionFeat = training_item['criterionFeat']
    logger = training_item['logger']
    logger.epoch_losses = {}  # 初始化 epoch_losses 字典
    
    
    
    for batch in tqdm(train_loader):
        real = batch['real_tensor'].to(device)
        label = batch['label_tensor'].to(device)
        filename = batch['filename']
        # zero grad
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        # test 
        # create one-hot vector for label map 
        input_label = label
        
        # fake image gerneration
        fake_image = netG(input_label)

        # fake image detection loss
        input_fake = torch.cat((input_label, fake_image.detach()), dim=1)
        pred_fake_pool = netD.forward(input_fake)
        loss_D_fake = criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss
        input_real = torch.cat((input_label, real.detach()), dim=1)
        pred_real = netD.forward(input_real)
        loss_D_real = criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)    
        input_g = torch.cat((input_label, fake_image), dim=1)    
        pred_fake = netD.forward(input_g)
        loss_G_fake = criterionGAN(pred_fake, True)     

        # feat loss
        loss_G_GAN_Feat = criterionFeat(pred_fake, pred_real)
        
                    
        # backward
        loss_D = loss_D_fake + loss_D_real*0.5
        loss_G = loss_G_fake + loss_G_GAN_Feat
        loss_D.backward()
        loss_G.backward()

        optimizer_D.step()
        optimizer_G.step()

        # 累加損失
        logger.update_epoch_losses({
            'Discriminator Fake Loss': loss_D_fake.item(),
            'Discriminator Real Loss': loss_D_real.item(),
            'Discriminator Loss': loss_D.item(),
            'Generator Fake Loss': loss_G_fake.item(),
            'Generator Feature Loss': loss_G_GAN_Feat.item(),
            'Generator Loss': loss_G.item()
        })

        if epoch % 20 == 0 :
            save_dir = os.path.join(args.project_dir, 'visualization/fake')
            save_tensor_2(fake_image.detach(), save_dir, f'{epoch}_{filename}.bmp')

    logger.log_epoch_losses(epoch)
    
    return{
        'netG': netG,
        'netD': netD,
        'logger': logger
    }


if __name__ == '__main__':
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    arg_parser = ArgumentParserWrapper()
    args = arg_parser.parse_args()


    training_config = initial(args)
    for epoch in range(args.epoch):
        train_result = train(training_config, epoch)

        if epoch % 20 == 0:
            ckpt_dir = os.path.join(args.project_dir, 'ckpt')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'epoch' : epoch,
                'generator' : train_result['netG'].state_dict(),
                'discriminator' : train_result['netD'].state_dict(),
            }, ckpt_dir+f'/epoch_{epoch}.pt')



