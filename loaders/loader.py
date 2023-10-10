import random
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
import shutil
import os
from utils.tools import get_name


def get_train_transformations():
    aug_list = [
                transforms.Resize((224, 224), Image.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomApply([transforms.RandomRotation([-90, 90])], p=0.8),
                # transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.15, 0.15))],
                #                        p=0.8),
                transforms.ToTensor(),
                ]
    return transforms.Compose(aug_list)


def get_val_transformations():
    aug_list = [
                transforms.Resize((224, 224), Image.BILINEAR),
                transforms.ToTensor(),
                ]
    return transforms.Compose(aug_list)


class MakeListImage():
    """
    this class used to make list of data for dataset
    """
    def __init__(self, types):
        self.types = types
        self.image_root = "data/concrete_cropped_center/raw"
        self.cat = [['bg'],
                    ['ひびわれ'],
                    ['剥離・鉄筋露出', '剥離・鉄筋露出(剥離のみ)'],
                    ['遊離石灰(つらら状)', '遊離石灰']]
        self.ratio = {0: 0.03, 2: 0.3}
        self.cats = {}

        for i in range(len(self.cat)):
            self.cats.update({i: []})

        all_folders = get_name(os.path.join(self.image_root))
        train, val = train_test_split(all_folders, train_size=0.80, random_state=15)
        print("class number: ", len(self.cat))
        print(len(train))
        print(len(val))
        self.all_folders = {"train": train, "val": val}

    def get_data(self):
        record_all = []
        folders = self.all_folders[self.types]

        for item in folders:
            imgs = get_name(os.path.join(self.image_root, item), mode_folder=False)

            for img in imgs:
                current_cat = img.split("_")[0]
                current_index = None
                for s in range(len(self.cat)):
                    if current_cat in self.cat[s]:
                        current_index = s

                if current_index is None:
                    continue

                self.cats[current_index].append(os.path.join(self.image_root, item, img))

        for key, value in self.cats.items():
            index = key
            if key in list(self.ratio.keys()):
                for_use, _ = train_test_split(value, train_size=self.ratio[key], random_state=1)
            else:
                for_use = value

            print(index, len(for_use))

            for terms in for_use:
                record_all.append([terms, int(index)])

        random.seed(43)
        random.shuffle(record_all)

        return record_all


class GenerateImages(torch.utils.data.Dataset):
    def __init__(self, types, transform=None):
        self.all_data = MakeListImage(types).get_data()
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item_id):
        image_root = self.all_data[item_id][0]
        image = Image.open(image_root).convert('RGB')
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.all_data[item_id][1]
        label = torch.from_numpy(np.array(label))
        return image, label


def loader_generation(args):
    transform_train = get_train_transformations()
    transform_val = get_val_transformations()

    train_set = GenerateImages("train", transform_train)
    val_set = GenerateImages("val", transform_val)
    print('Train samples %d - Val samples %d' % (len(train_set), len(val_set)))

    train_loader1 = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=False, drop_last=True)
    train_loader2 = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=False, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=args.num_workers,
                           pin_memory=False, drop_last=False)
    return train_loader1, train_loader2, val_loader


def load_all_imgs(args):
    def filter(data):
        imgs = []
        labels = []
        for i in range(len(data)):
            root = data[i][0]
            ll = int(data[i][1])
            imgs.append(root)
            labels.append(ll)
        return imgs, labels

    train = MakeListImage("train").get_data()
    val = MakeListImage("val").get_data()
    cat = MakeListImage("val").cat
    train_imgs, train_labels = filter(train)
    val_imgs, val_labels = filter(val)
    return train_imgs, train_labels, val_imgs, val_labels, cat
