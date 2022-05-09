from data.randaugment import RandAugmentMC
from data.augmix import augmix
from torchvision import transforms
import numpy as np

class TransformWSW(object):
    def __init__(self, mean, std, size_image=32,strong_type = 'randaugment'):
        self.type = strong_type
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.rand_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def control(self,mode):
        assert (mode in ['100','010','001','110','011','101','111']),f'error mode - {mode} !!!'
        self.mode = mode
    
    def __call__(self, x):
        mode = int(self.mode,base = 2)
        augmentations = []
        if mode & 0x04 != 0:
            weak_plus = self.normalize(self.weak(x))
            augmentations.append(weak_plus)
        if mode & 0x02 != 0:
            if self.type == 'randaugment':
                strong = self.normalize(self.rand_augment(x))
            elif self.type == 'augmix':
                print(type(x))
                strong = self.normalize(augmix(np.array(x)))
            else:
                raise AssertionError(f'unknown strong augmentation type {self.type}')
            augmentations.append(strong)
        if mode & 0x01 != 0:
            weak = self.normalize(self.weak2(x))
            augmentations.append(weak)
        if len(augmentations) == 1:
            return augmentations[0]
        else:
            return augmentations
