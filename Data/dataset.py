import torch
from torch.utils import data
from torchvision import transforms
from Data.color_data import GrayscaleImageFolder


def create_train_loader(args):
    train_transforms = transforms.Compose([transforms.Resize((256, 256)),transforms.RandomHorizontalFlip()])
    train_imagefolder = GrayscaleImageFolder(args.train_path, train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=args.batch_size, shuffle=True)

    return train_imagefolder, train_loader


def create_test_loader(args):
    test_transforms = transforms.Compose([transforms.Resize((256, 256))])
    test_imagefolder = GrayscaleImageFolder(args.test_path, test_transforms)
    test_loader = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=False)

    return test_imagefolder, test_loader


def create_valid_loader(args):
    valid_transforms = transforms.Compose([transforms.Resize((256, 256))])
    valid_imagefolder = GrayscaleImageFolder(args.valid_path, valid_transforms)
    valid_loader = torch.utils.data.DataLoader(valid_imagefolder, batch_size=1, shuffle=False)

    return valid_imagefolder, valid_loader
