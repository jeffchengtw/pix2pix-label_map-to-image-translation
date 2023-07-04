
import torch
from tqdm import tqdm
from utils.visualization import*
from logger.loss_logger import*
from option.base import*


def update_learning_rate(opt, niter_decay, optimizer_G, optimizer_D):
    lrd = opt.lr / niter_decay
    lr = optimizer_G.param_groups[0]['lr'] - lrd        
    
    for param_group in optimizer_D.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_G.param_groups:
        param_group['lr'] = lr
    if opt.debug_mode:
        print('update learning rate: %f -> %f' % (old_lr, lr))
    old_lr = lr


def train(args, training_item, epoch):
    device = training_item['device']
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
            save_tensor_2(fake_image.detach(), save_dir, filename, epoch)
    
    
    logger.log_learning_rates({
        'learning rate D': optimizer_D.param_groups[0]['lr'], 
        'learning rate G': optimizer_G.param_groups[0]['lr']},
        epoch
        )

    ### linearly decay learning rate after certain iterations
    if epoch > (args.epoch * args.decay_rate):
       update_learning_rate(
           opt=args, 
           niter_decay=args.epoch * (1-args.decay_rate),
           optimizer_G=optimizer_G, 
           optimizer_D=optimizer_D
        )

    logger.log_epoch_losses(epoch)
    
    return{
        'netG': netG,
        'netD': netD,
        'logger': logger
    }