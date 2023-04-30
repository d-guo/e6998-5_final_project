# Module containing networks and methods for GANs

import torch


class Generator(torch.nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        # 3 * 28 * 140
        self.convblock1 = ConvBlock(
            channels=(3, 8),
            filter_shape=(3, 3),
            stride=1,
            padding=0,
            batch_norm=False,
            activation=True,
        )
        # 8 * 26 * 138
        self.convblock2 = ConvBlock(
            channels=(8, 16),
            filter_shape=(3, 3),
            stride=(2, 4),
            padding=0,
            batch_norm=True,
            activation=True,
        )
        # 16 * 12 * 34
        self.convblock3 = ConvBlock(
            channels=(16, 32),
            filter_shape=(3, 3),
            stride=(2, 4),
            padding=0,
            batch_norm=True,
            activation=True,
        )
        # 32 * 5 * 8
        self.resblock1 = ResBlock(32)
        self.resblock2 = ResBlock(32)
        self.resblock3 = ResBlock(32)
        # 32 * 5 * 8
        self.transconvblock1 = TransConvBlock(
            channels=(32, 16),
            filter_shape=(3, 3),
            stride=(2, 4),
            output_padding=(1, 3),
            batch_norm=True,
            activation=True,
        )
        # 16 * 12 * 34
        self.transconvblock2 = TransConvBlock(
            channels=(16, 8),
            filter_shape=(3, 3),
            stride=(2, 4),
            output_padding=(1, 3),
            batch_norm=True,
            activation=True,
        )
        # 8 * 26 * 138
        self.transconvblock3 = TransConvBlock(
            channels=(8, 3),
            filter_shape=(3, 3),
            stride=1,
            output_padding=0,
            batch_norm=True,
            activation=True,
        )
        # 3 * 28 * 140
        
    def forward(self, inp):
        """
        input should be tensor with dimensions (num_samples, C, M, N)
        """
        
        out = self.convblock1(inp)
        out = self.convblock2(out)
        out = self.convblock3(out)
        
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        
        out = self.transconvblock1(out)
        out = self.transconvblock2(out)
        out = self.transconvblock3(out)

        return out
    
class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        # 3 * 28 * 140
        self.convblock1 = ConvBlock(
            channels=(3, 8),
            filter_shape=(4, 4),
            stride=1,
            padding=0,
            batch_norm=False,
            activation=True,
        )
        # 8 * 25 * 137
        self.convblock2 = ConvBlock(
            channels=(8, 16),
            filter_shape=(4, 4),
            stride=(2, 4),
            padding=0,
            batch_norm=True,
            activation=True,
        )
        # 16 * 11 * 34
        self.convblock3 = ConvBlock(
            channels=(16, 32),
            filter_shape=(4, 4),
            stride=(2, 4),
            padding=0,
            batch_norm=True,
            activation=True,
        )
        # 32 * 4 * 8
        self.convblock4 = ConvBlock(
            channels=(32, 1),
            filter_shape=(4, 8),
            stride=1,
            padding=0,
            batch_norm=False,
            activation=False,
        )
        # 1 * 1 * 1
        self.sigmoid1 = torch.nn.Sigmoid()
        # 1 * 1 * 1

    def forward(self, inp):
        """
        input should be tensor with dimensions (num_samples, C, M, N)
        """
        
        num_samples = inp.shape[0]
        C = inp.shape[1]
        
        out = self.convblock1(inp)
        out = self.convblock2(out)
        out = self.convblock3(out)
        out = self.convblock4(out)
        
        out = out.reshape(num_samples, 1)
        
        out = self.sigmoid1(out)
        
        return out
    
class ConvBlock(torch.nn.Module):
    
    def __init__(self, channels, filter_shape, stride, padding, batch_norm, activation):
        super(ConvBlock, self).__init__()
        
        self.conv = torch.nn.Conv2d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=filter_shape,
            stride=stride,
            padding=padding,
        )
        
        if batch_norm:
            self.batchnorm = torch.nn.BatchNorm2d(
                num_features=channels[1],
            )
        else:
            self.batchnorm = torch.nn.Identity()
            
        if activation:
            self.act = torch.nn.LeakyReLU()
        else:
            self.act = torch.nn.Identity()
        
    def forward(self, inp):
        """
        input should be tensor with dimensions (num_samples, C, N, N)
        """
        
        out = self.conv(inp)
        out = self.batchnorm(out)
        out = self.act(out)
        
        return out
    
class TransConvBlock(torch.nn.Module):
    
    def __init__(self, channels, filter_shape, stride, output_padding, batch_norm, activation):
        super(TransConvBlock, self).__init__()
        
        self.transconv = torch.nn.ConvTranspose2d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=filter_shape,
            stride=stride,
            output_padding=output_padding,
        )
        
        if batch_norm:
            self.batchnorm = torch.nn.BatchNorm2d(
                num_features=channels[1],
            )
        else:
            self.batchnorm = torch.nn.Identity()
            
        if activation:
            self.act = torch.nn.LeakyReLU()
        else:
            self.act = torch.nn.Identity()
        
    def forward(self, inp):
        """
        input should be tensor with dimensions (num_samples, C, N, N)
        """
        
        out = self.transconv(inp)
        out = self.batchnorm(out)
        out = self.act(out)
        
        return out
    
class ResBlock(torch.nn.Module):
    
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        
        self.convblock1 = ConvBlock(
            channels=(channels, channels),
            filter_shape=(3, 3),
            stride=1,
            padding=1,
            batch_norm=True,
            activation=True,
        )
        self.convblock2 = ConvBlock(
            channels=(channels, channels),
            filter_shape=(3, 3),
            stride=1,
            padding=1,
            batch_norm=True,
            activation=True,
        )
        
    def forward(self, inp):
        """
        input should be tensor with dimensions (num_samples, C, N, N)
        """
        
        out = self.convblock1(inp)
        out = self.convblock2(out)
        out = out + inp
        
        return out

        
def compute_discr_loss_minimax(discr_model, target_samples, gen_samples):
    """
    minimax loss for discriminator
    """
    
    num_samples = target_samples.shape[0]
    
    discr_target_preds = discr_model(target_samples)
    discr_gen_preds = discr_model(gen_samples)
    
    J_D = -torch.sum(torch.log(discr_target_preds)) / num_samples - torch.sum(torch.log(1 - discr_gen_preds)) / num_samples
    
    return J_D;

def compute_gen_loss_minimax(discr_model, target_samples, gen_samples):
    """
    minimax loss for generator
    """
    
    num_samples = target_samples.shape[0]
    
    discr_target_preds = discr_model(target_samples)
    discr_gen_preds = discr_model(gen_samples)
    
    J_G = torch.sum(torch.log(discr_target_preds)) / num_samples + torch.sum(torch.log(1 - discr_gen_preds)) / num_samples
    
    return J_G;

def compute_gen_loss_minimax_modified(discr_model, gen_samples):
    """
    modified minimax loss for generator
    """
    
    num_samples = gen_samples.shape[0]

    discr_gen_preds = discr_model(gen_samples)
    
    J_G = -torch.sum(torch.log(discr_gen_preds)) / num_samples;
    
    return J_G;