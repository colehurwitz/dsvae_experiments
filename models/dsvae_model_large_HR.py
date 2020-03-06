import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

def truncated_normal_(tensor, mean=0, std=0.02):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        with torch.no_grad():
            truncated_normal_(m.weight.data, mean=0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
class DSVAELHR(nn.Module):
    def __init__(self, z_dim, device=None):
        super(DSVAELHR, self).__init__()
        self.z_dim = z_dim
        if device is None:
            self.cuda = False
            self.device = None
        else:
            self.device = device
            self.cuda = True
        
        #ENCODER RESIDUAL
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 64, 64]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 32, 32]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 16, 16]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 8, 8]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)

        self.fc1 = nn.Linear(512*8*8, 256)
        weights_init(self.fc1)
        
        self.fc_mean = nn.Linear(256, z_dim)
        weights_init(self.fc_mean)
        self.fc_var = nn.Linear(256, z_dim)
        weights_init(self.fc_var)

        #DECODER
        self.d1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 64, 64]
        weights_init(self.d1)

        self.d2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 32, 32]
        weights_init(self.d2)
        self.mu2 = nn.Linear(self.z_dim, 128*32*32)
        self.sig2 = nn.Linear(self.z_dim, 128*32*32)
        self.instance_norm_d2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.d3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 16, 16]
        weights_init(self.d3)
        self.mu3 = nn.Linear(self.z_dim, 256*16*16)
        self.sig3 = nn.Linear(self.z_dim, 256*16*16)
        self.instance_norm_d3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.d4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 8, 8]
        weights_init(self.d4)
        self.mu4 = nn.Linear(self.z_dim, 512*8*8)
        self.sig4 = nn.Linear(self.z_dim, 512*8*8)
        self.instance_norm_d4 = nn.InstanceNorm2d(num_features=512, affine=False)
        
        self.d4_5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 4, 4]
        weights_init(self.d4_5)
        self.mu4_5 = nn.Linear(self.z_dim, 512*4*4)
        self.sig4_5 = nn.Linear(self.z_dim, 512*4*4)
        self.instance_norm_d4_5 = nn.InstanceNorm2d(num_features=512, affine=False)
        
        self.fc2 = nn.Linear(512*4*4, 512*4*4)
        weights_init(self.fc2)
        
        self.d5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) #[b, 256, 8, 8]
        weights_init(self.d5)
        self.mu5 = nn.Linear(self.z_dim, 256*8*8)
        self.sig5 = nn.Linear(self.z_dim, 256*8*8)
        self.instance_norm_d5 = nn.InstanceNorm2d(num_features=256, affine=False) 
        
        self.d6 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) #[b, 128, 16, 16]
        weights_init(self.d6)
        self.mu6 = nn.Linear(self.z_dim, 128*16*16)
        self.sig6 = nn.Linear(self.z_dim, 128*16*16)
        self.instance_norm_d6 = nn.InstanceNorm2d(num_features=128, affine=False) 
        
        self.d7 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) #[b, 64, 32, 32]
        weights_init(self.d7)
        self.mu7 = nn.Linear(self.z_dim, 64*32*32)
        self.sig7 = nn.Linear(self.z_dim, 64*32*32)
        self.instance_norm_d7 = nn.InstanceNorm2d(num_features=64, affine=False) 
        
        self.d8 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) #[b, 32, 64, 64]
        weights_init(self.d8)
        self.mu8 = nn.Linear(self.z_dim, 32*64*64)
        self.sig8 = nn.Linear(self.z_dim, 32*64*64)
        self.instance_norm_d8 = nn.InstanceNorm2d(num_features=32, affine=False) 
        
        self.d9 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1) #[b, 3, 128, 128]
        weights_init(self.d9)
                
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))   #[b, 64, 64, 64]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))     #[b, 128, 32, 32]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))     #[b, 256, 16, 16]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))     #[b, 512, 8, 8]
        h = self.leakyrelu(self.fc1(h.view(-1,512*8*8)))           #[b, 512*8*8]
        return self.fc_mean(h), F.softplus(self.fc_var(h))        #[b, z_dim]

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 

    def decode(self, y, z):
        h = self.leakyrelu(self.d1(y)) #[b, 64, 32, 32]
        
        mu = self.mu2(z).reshape(-1, 128, 32, 32)
        sig = self.sig2(z).reshape(-1, 128, 32, 32)
        h = self.leakyrelu(sig*self.instance_norm_d2(self.d2(h)) + mu) #[b, 128, 32, 32]
        
        mu = self.mu3(z).reshape(-1, 256, 16, 16)
        sig = self.sig3(z).reshape(-1, 256, 16, 16)
        h = self.leakyrelu(sig*self.instance_norm_d3(self.d3(h)) + mu) #[b, 256, 16, 16]
        
        mu = self.mu4(z).reshape(-1, 512, 8, 8)
        sig = self.sig4(z).reshape(-1, 512, 8, 8)
        h = self.leakyrelu(sig*self.instance_norm_d4(self.d4(h)) + mu) #[b, 512, 8, 8]
        
        mu = self.mu4_5(z).reshape(-1, 512, 4, 4)
        sig = self.sig4_5(z).reshape(-1, 512, 4, 4)
        h = self.leakyrelu(sig*self.instance_norm_d4_5(self.d4_5(h)) + mu) #[b, 512, 4, 4]
       
        h = self.fc2(h.view(-1,512*4*4))                                #[b, 512*4*4]
        h = h.reshape(-1, 512, 4, 4)                                    #[b, 512, 4, 4]
        
        mu = self.mu5(z).reshape(-1, 256, 8, 8)
        sig = self.sig5(z).reshape(-1, 256, 8, 8)
        h = self.leakyrelu(sig*self.instance_norm_d5(self.d5(h)) + mu) #[b, 256, 8, 8]
        
        mu = self.mu6(z).reshape(-1, 128, 16, 16)
        sig = self.sig6(z).reshape(-1, 128, 16, 16)
        h = self.leakyrelu(sig*self.instance_norm_d6(self.d6(h)) + mu) #[b, 128, 16, 16]
        
        mu = self.mu7(z).reshape(-1, 64, 32, 32)
        sig = self.sig7(z).reshape(-1, 64, 32, 32)
        h = self.leakyrelu(sig*self.instance_norm_d7(self.d7(h)) + mu) #[b, 64, 32, 32]
        
        mu = self.mu8(z).reshape(-1, 32, 64, 64)
        sig = self.sig8(z).reshape(-1, 32, 64, 64)
        h = self.leakyrelu(sig*self.instance_norm_d8(self.d8(h)) + mu) #[b, 32, 64, 64]
        
        return self.sigmoid(self.d9(h))                                #[b, 3, 128, 128]

    def forward(self, y, x):
        mu, var = self.encode(x)
        if self.training:
            z = self.reparametrize(mu, var)
        else:
            z = mu
        x_hr_hat = self.decode(y, z)
        return x_hr_hat, mu, var

reconstruction_function = nn.BCELoss(reduction='sum')
def loss_function(x_hr_hat, x_hr, mu, var):
    BCE = reconstruction_function(x_hr_hat, x_hr)/x_hr_hat.shape[0]

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    logvar = torch.log(var)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)/x_hr_hat.shape[0] 

    return BCE + KLD, BCE, KLD