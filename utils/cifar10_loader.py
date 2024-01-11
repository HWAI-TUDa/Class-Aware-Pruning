from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random

MEAN = (0.49154913, 0.4821251, 0.44642678)
STD = (0.24703223, 0.24348513, 0.26158784)

# MEAN = (0.485, 0.456, 0.406) 
# STD  = (0.229, 0.224, 0.225)

normalize = transforms.Normalize(mean=MEAN, std=STD)
def loader(batch_size=256, num_workers=2, pin_memory=True):
    return DataLoader(
          datasets.CIFAR10(root='./data', train=True, download=False,
                    transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, 4),
                            transforms.ToTensor(),
                            normalize,
                            ])),
          batch_size=batch_size,
          shuffle=True,
          num_workers=num_workers,
          pin_memory=pin_memory)
def test_loader(batch_size=512, num_workers=2, pin_memory=True):
    return DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=False,
                       transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)

def classes_loader(class_idx = 0, batch_size=256, num_workers=2, pin_memory=True):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    classes_set = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
           
    class_indices = [i for i, (_, label) in enumerate(classes_set) if label == class_idx]
    trainset_class = Subset(classes_set, class_indices)

    sub_trainset_size = len(trainset_class) // 500
    
    random_indices = random.sample(range(len(trainset_class)), sub_trainset_size)
    trainset_class = Subset(trainset_class, random_indices)

    return DataLoader(trainset_class, batch_size = batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      pin_memory=pin_memory)  
