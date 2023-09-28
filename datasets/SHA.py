import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
import torchvision.transforms as standard_transforms
import warnings
warnings.filterwarnings('ignore')

class SHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, flip=False):
        self.root_path = data_root
        
        prefix = "train_data" if train else "test_data"
        self.prefix = prefix
        self.img_list = os.listdir(f"{data_root}/{prefix}/images")

        # get image and ground-truth list
        self.gt_list = {}
        for img_name in self.img_list:
            img_path = f"{data_root}/{prefix}/images/{img_name}"  
            gt_path = f"{data_root}/{prefix}/ground-truth/GT_{img_name}"
            self.gt_list[img_path] = gt_path.replace("jpg", "mat")
        self.img_list = sorted(list(self.gt_list.keys()))
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = 256
    
    def compute_density(self, points):
        """
        Compute crowd density:
            - defined as the average nearest distance between ground-truth points
        """
        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:,1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # load image and gt points
        img_path = self.img_list[index]
        gt_path = self.gt_list[img_path]
        img, points = load_data((img_path, gt_path), self.train)
        points = points.astype(float)

        # image transform
        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)

        # random scale
        if self.train:
            scale_range = [0.8, 1.2]           
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            
            # interpolation
            if scale * min_size > self.patch_size:  
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                points *= scale

        # random crop patch
        if self.train:
            img, points = random_crop(img, points, patch_size=self.patch_size)

        # random flip
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            points[:, 1] = self.patch_size - points[:, 1]

        # target
        target = {}
        target['points'] = torch.Tensor(points)
        target['labels'] = torch.ones([points.shape[0]]).long()

        if self.train:
            density = self.compute_density(points)
            target['density'] = density

        if not self.train:
            target['image_path'] = img_path

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = io.loadmat(gt_path)['image_info'][0][0][0][0][0][:,::-1]
    return img, points


def random_crop(img, points, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size
    
    # random crop
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w
    idx = (points[:, 0] >= start_h) & (points[:, 0] <= end_h) & (points[:, 1] >= start_w) & (points[:, 1] <= end_w)

    # clip image and points
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_points = points[idx]
    result_points[:, 0] -= start_h
    result_points[:, 1] -= start_w
    
    # resize to patchsize
    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h/imgH, patch_w/imgW
    result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
    result_points[:, 0] *= fH
    result_points[:, 1] *= fW
    return result_img, result_points


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    
    data_root = args.data_path
    if image_set == 'train':
        train_set = SHA(data_root, train=True, transform=transform, flip=True)
        return train_set
    elif image_set == 'val':
        val_set = SHA(data_root, train=False, transform=transform)
        return val_set
