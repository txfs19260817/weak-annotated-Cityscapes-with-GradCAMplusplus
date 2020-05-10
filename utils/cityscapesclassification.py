import os
import torch
import torchvision.transforms as transforms
from PIL import Image

import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class CityscapesClassificationDataset(torch.utils.data.Dataset):
    """List your dataset according to the following manner:
    root
    └─cityscapesclassification
        ├─gt
        │  ├─train
        │  ├─train_extra
        │  └─val
        ├─leftImg8bit
        │  ├─train
        │  ├─train_extra
        │  └─val
        └─lists
    """

    def __init__(self, root=os.path.expanduser('~/.encoding/data'), transform=None,
                 target_transform=None, split='train', **kwargs):

        super(CityscapesClassificationDataset, self).__init__()

        self.nclass = 19

        # split
        assert split in ('train', 'train_extra', 'val')
        self.split = split

        # paths
        self.BASE_DIR = "cityscapesclassification"
        self.root = os.path.join(root, self.BASE_DIR)
        self.img_folder = os.path.join(self.root, 'leftImg8bit', self.split)
        self.gt_folder = os.path.join(self.root, 'gt', self.split)
        self.list_path = os.path.join(self.root, 'lists', self.split + '_id.txt')

        # filename
        with open(self.list_path, 'r') as f:
            self.filename_list = f.read().splitlines()

        # transform init
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.target_transform = target_transform  # no use

    def __getitem__(self, index):

        # paths
        filename = self.filename_list[index]
        img_path = os.path.join(self.img_folder, filename + '.png')
        lbl_path = os.path.join(self.gt_folder, filename + '.txt')

        # read
        image = Image.open(img_path).convert('RGB')
        with open(lbl_path, 'r') as f:
            label = f.readline().rstrip('\r\n')

        # transform
        if self.transform is not None:
            image = self.transform(image)

        label = list(map(int, label.split(',')))
        label = torch.FloatTensor(label)

        return image, label, filename + '.png'

    def __len__(self):

        return len(self.filename_list)


if __name__ == "__main__":

    # 1. fundamental test
    trainset = CityscapesClassificationDataset(root='E:/')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True)
    valset = CityscapesClassificationDataset(root='E:/', split='val')
    valloader = torch.utils.data.DataLoader(valset, batch_size=2, shuffle=False)
    train_extraset = CityscapesClassificationDataset(root='E:/', split='train_extra')
    train_extraloader = torch.utils.data.DataLoader(train_extraset, batch_size=2, shuffle=True)

    for batch_idx, (data, target) in enumerate(trainloader):
        break

    print(len(trainloader), len(valloader), len(train_extraloader))
    print(batch_idx, (data, target))

    # 2. concat datasets test

    trainset = torch.utils.data.ConcatDataset([trainset, train_extraset])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True)

    print(len(trainloader))
