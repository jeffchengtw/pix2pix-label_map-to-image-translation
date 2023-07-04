import torch
import torch.optim as optim
from tqdm import tqdm
from dataset import MyDataset
from train import train
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
        input_nc=input_nc,
        target_size=(input_w, input_h)
    )
    args.label_nc = train_dataset.label_nc
    train_loader = DataLoader(train_dataset, batch_size=8)
    

    netG = Generator(
        input_nc=args.label_nc, 
        output_nc=output_nc, 
        n_downsampling=n_downsampling, 
        cfg=args
    ).to(device)
    
    netD = MultiscaleDiscriminator(
        input_nc=(output_nc+args.label_nc),
        num_D=num_D, 
        n_layers=n_layer_D
    ).to(device)

    criterionGAN = GANLoss(use_lsgan=True).to(device)
    criterionFeat = FeatureLoss(num_D=num_D, n_layers_D=n_layer_D)

    optimizer_G = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.999))

    logger = Logger(log_dir=os.path.join(args.project_dir, 'logs'))

    return {
        'device': device,
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



if __name__ == '__main__':
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    arg_parser = ArgumentParserWrapper()
    args = arg_parser.parse_args()

    training_config = initial(args)
    for epoch in range(args.epoch):
        train_result = train(args, training_config, epoch)

        if epoch % 20 == 0:
            ckpt_dir = os.path.join(args.project_dir, 'ckpt')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'epoch' : epoch,
                'config' : args,
                'generator' : train_result['netG'].state_dict(),
                'discriminator' : train_result['netD'].state_dict(),
            }, ckpt_dir+f'/epoch_{epoch}.pt')
    
    print('End !!!')
    print('logs dir : tensorboard --logdir= ', os.path.join(args.project_dir, 'logs'))



