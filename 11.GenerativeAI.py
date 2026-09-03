import torch 
import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, latent_dim=100, img_dim=784):
    super(Generator, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(latent_dim, 256),
      nn.LeakyReLU(0.2), nn.BatchNorm1d(256),
      nn.Linear(256,512),
      nn.LeakyReLU(0.2), nn.BatchNorm1d(512),
      nn.Linear(512, img_dim), 
      nn.Tanh()
    )
  def forward(self, z): return self.model(z)

class Discriminator(nn.Module):
  def __init__ (self, img_dim=784);
    super(Discriminator, self).__init__()
    self.model = nn.Sequential(
      nn.Linear(img_dim, 512),
      
    )
