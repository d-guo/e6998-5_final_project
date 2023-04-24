# Module containing networks and methods for GANs

import torch


class Generator(torch.nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
#         self.convblock1 = ConvBlock(
#             channels=(1, 16),
#             filter_shape=(3, 3),
#             downsample=True,
#             batch_norm=False,
#             activation=True,
#         )
        
#         self.convblock2 = ConvBlock(
#             channels=(16, 32),
#             filter_shape=(3, 3),
#             downsample=True,
#             batch_norm=True,
#             activation=True,
#         )
        
#         self.convblock3 = ConvBlock(
#             channels=(32, 64),
#             filter_shape=(3, 3),
#             downsample=True,
#             batch_norm=True,
#             activation=True,
#         )
        
#         self.transconvblock1 = TransConvBlock(
#             channels=(64, 32),
#             filter_shape=(3, 3),
#             upsample=True,
#             batch_norm=True,
#             activation=True,
#         )
        
#         self.transconvblock2 = TransConvBlock(
#             channels=(32, 16),
#             filter_shape=(3, 3),
#             upsample=True,
#             batch_norm=True,
#             activation=True,
#         )
        
#         self.transconvblock3 = TransConvBlock(
#             channels=(16, 1),
#             filter_shape=(3, 3),
#             upsample=True,
#             batch_norm=False,
#             activation=True,
#         )

        self.test = torch.nn.Linear(1 * 1 * 28 * 5 * 28, 1 * 1 * 28 * 5 * 28)
        
    def forward(self, inp):
        """
        input should be tensor with dimensions (num_samples, 1, M, N)
        """
        
#         out = self.convblock1(inp)
#         out = self.convblock2(out)
#         out = self.convblock3(out)
#         out = self.transconvblock1(out)
#         out = self.transconvblock2(out)
#         out = self.transconvblock3(out)

        out = inp.reshape(1 * 1 * 28 * 5 * 28)
        out = self.test(out)
        out = out.reshape(1, 1, 28, 28 * 5)
        
        return out
    
class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.convblock1 = ConvBlock(
            channels=(1, 16),
            filter_shape=(3, 3),
            downsample=True,
            batch_norm=False,
            activation=True,
        )
        
        self.convblock2 = ConvBlock(
            channels=(16, 32),
            filter_shape=(3, 3),
            downsample=True,
            batch_norm=True,
            activation=True,
        )
        
        self.convblock3 = ConvBlock(
            channels=(32, 64),
            filter_shape=(3, 3),
            downsample=True,
            batch_norm=True,
            activation=True,
        )
        
        self.convblock4 = ConvBlock(
            channels=(64, 128),
            filter_shape=(3, 3),
            downsample=True,
            batch_norm=False,
            activation=True,
        )
        
        self.fc1 = torch.nn.Linear(128 * 25 * 45, 64)
        self.act1 = torch.nn.LeakyReLU()
        
        self.fc2 = torch.nn.Linear(64, 1)
        
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, inp):
        """
        input should be tensor with dimensions (num_samples, 1, M, N)
        """
        
        num_samples = inp.shape[0]
        
        out = self.convblock1(inp)
        out = self.convblock2(out)
        out = self.convblock3(out)
        out = self.convblock4(out)
        
        out = out.reshape(num_samples, 128, 25, 45)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.fc2(out)
        
        out = self.sigmoid1(out)
        
        return out
    
class ConvBlock(torch.nn.Module):
    
    def __init__(self, channels, filter_shape, downsample, batch_norm, activation):
        super(ConvBlock, self).__init__()
        
        self.conv = torch.nn.Conv2d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=filter_shape,
            stride=2 if downsample else 1,
            padding='valid' if downsample else 'same',
        
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
    
    def __init__(self, channels, filter_shape, upsample, batch_norm, activation):
        super(TransConvBlock, self).__init__()
        
        self.transconv = torch.nn.ConvTranspose2d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=filter_shape,
            stride=2 if upsample else 1,
            padding=2,
        
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