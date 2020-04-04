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
        
class DSVAE_ADD(nn.Module):
    def __init__(self, z_dim, y_shape, device=None):
        super(DSVAE_ADD, self).__init__()
        self.z_dim = z_dim
        
        self.y_shape = y_shape
        self.y_nc = y_shape[0]
        self.y_height = y_shape[1]
        self.y_width = y_shape[2]
        
        if device is None or device == -1:
            self.cuda = False
            self.device = None
        else:
            self.device = device
            self.cuda = True
            print(device)
        
        #ENCODER RESIDUAL
        self.e1 = nn.Conv2d(self.y_nc, 32, 4, stride=2, padding=1)  #[b,64,inh/2,inw/2]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=32, affine=False)

        self.e2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  #[b,128,inh/4,inw/4]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  #[b,256,inh/8,inw/8]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.e4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)  #[b,512,inh/16,inw/16]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=256, affine=False)
        
        self.fc_mean = nn.Linear(256*int(self.y_height/16)*int(self.y_width/16), z_dim)
        weights_init(self.fc_mean)
        self.fc_var = nn.Linear(256*int(self.y_height/16)*int(self.y_width/16), z_dim)
        weights_init(self.fc_var)

        #DECODER
        self.fc1 = nn.Linear(z_dim, 256*int(self.y_height/16)*int(self.y_width/16))
        weights_init(self.fc1)
        
        self.d1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1) #[b, 256,yh/8,yw/8]
        weights_init(self.d1)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=128, affine=False) 
        
        self.d2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1) #[b, 128,yh/4,yw/4]
        weights_init(self.d2)
        self.instance_norm_d2 = nn.InstanceNorm2d(num_features=64, affine=False) 
        
        self.d3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1) #[b, 64,yh/2,yw/2]
        weights_init(self.d3)
        self.instance_norm_d3 = nn.InstanceNorm2d(num_features=32, affine=False) 
        
        self.d4 = nn.ConvTranspose2d(32, self.y_nc, 4, stride=2, padding=1) #[b, ync,yh,yw]
        weights_init(self.d4)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, y, residual_method='subtract'):
        if residual_method == 'subtract':
            residual = x - y
        elif residual_method == 'none':
            residual = x
        else:
            raise ValueError("Other residual methods not implemented yet")
        h = self.leakyrelu(self.instance_norm_e1(self.e1(residual)))   
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))    
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))     
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))
        h = h.view(-1,256*int(self.y_height/16)*int(self.y_width/16))
        return self.fc_mean(h), F.softplus(self.fc_var(h))

    def reparametrize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 

    def decode(self, z, y):
        h = self.fc1(z)
        h = h.view(-1,256,int(self.y_height/16),int(self.y_width/16))
        h = self.relu(self.instance_norm_d1(self.d1(h)))   
        h = self.relu(self.instance_norm_d2(self.d2(h)))    
        h = self.relu(self.instance_norm_d3(self.d3(h)))     
        mask = self.d4(h)
#         x_hat = y + mask
#         return torch.clamp(x_hat, 0, 1)     
#         return torch.sigmoid(x_hat)
#         return  x_hat
        return mask
    
    def forward(self, x, y, residual_method='subtract'):
        mu, var = self.encode(x, y, residual_method)
        if self.training:
            z = self.reparametrize(mu, var)
        else:
            z = mu
#         x_hat = self.decode(z, y)
#         return x_hat, mu, var
        mask = self.decode(z, y)
        return mask, mu, var

# reconstruction_function = nn.BCELoss(reduction='sum')
# reconstruction_function = nn.BCELoss(reduction='sum')
def loss_function(x_hat, x, mu, var, reconstruction_function):
    recon_loss = reconstruction_function(x_hat, x)/x_hat.shape[0]
    
    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    logvar = torch.log(var)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)/x_hat.shape[0] 

    return recon_loss + KLD, recon_loss, KLD



