import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as T

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
mean = torch.tensor([0.4914, 0.4822, 0.4465])  # CIFAR-10 mean
std = torch.tensor([0.2023, 0.1994, 0.2010])   # CIFAR-10 std

class GaussianNoise(object):
  def __init__(self, mean=0., std=1.):
    self.std = std
    self.mean = mean

  def __call__(self, tensor):
    return tensor + torch.randn(tensor.size()) * self.std + self.mean

  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class MaskingNoise(object):
  def __init__(self, prob, mean=mean, std=std):
    mean=torch.tensor(mean).view(3,1,1)
    std=torch.tensor(std).view(3,1,1)
    self.prob = prob
    self.black = (0-mean)/std

  def __call__(self, tensor):
    mask = torch.rand(tensor.shape[-2:])
    return self.black * (mask<self.prob)+tensor*(mask>self.prob)

  def __repr__(self):
    return self.__class__.__name__ + '(prob={0})'.format(self.prob)

class SaltPepperNoise(object):
  def __init__(self, prob, mean=mean, std=std):
    mean=torch.tensor(mean).view(3,1,1)
    std=torch.tensor(std).view(3,1,1)
    self.prob = prob
    self.black = (0-mean)/std
    self.white = (1-mean)/std

  def __call__(self, tensor):
    mask = torch.rand(tensor.shape[-2:])
    mask2 = torch.logical_and(self.prob / 2 <= mask, mask <= (1-self.prob/2))
    return self.black * (mask<(self.prob/2)) + self.white * (mask>(1-self.prob/2)) + tensor*mask2

  def __repr__(self):
    return self.__class__.__name__ + '(prob={0})'.format(self.prob)

class NoisyDataset(dset.CIFAR100):
  def __init__(self, root, train=True, transform=transform, download=True, noise_model=None):
    super().__init__(root=root, train=train, transform=transform, download=download)
    self.noise_model = noise_model 

  def __getitem__(self, index):
    img, target = super().__getitem__(index)
    if self.noise_model is not None:
        noisy_img = self.noise_model(img)
    return noisy_img, img, target