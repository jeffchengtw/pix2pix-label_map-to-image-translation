import torch
import torch.nn as nn
from torch.autograd import Variable

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            print(target_tensor)
            return self.loss(input[-1], target_tensor)
        
class FeatureLoss(nn.Module):
    def __init__(self, num_D, n_layers_D):
        super(FeatureLoss, self).__init__()
        self.n_layers_D = n_layers_D
        self.num_D = num_D
        self.feat_weights = 4.0 / (self.n_layers_D + 1)
        self.D_weights = 1.0 / self.num_D
        self.criterionFeat = nn.L1Loss()

    def forward(self, pred_fake, pred_real):
        loss_G_GAN_Feat = 0
        for i in range(self.num_D):
            for j in range(len(pred_fake[i])-1):
                loss_G_GAN_Feat += self.D_weights * self.feat_weights * \
                    self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * 10
        return loss_G_GAN_Feat
    
if __name__ == '__main__':
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterionGAN = GANLoss(use_lsgan=True).to(device)

    input = torch.rand((1, 3, 32, 32))
    print(input)
    loss = criterionGAN(input, False)