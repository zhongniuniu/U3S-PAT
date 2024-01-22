import os
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def display_arr_stats(arr):
    shape, vmin, vmax, vmean, vstd = arr.shape, np.min(arr), np.max(arr), np.mean(arr), np.std(arr)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


def display_tensor_stats(tensor):
    shape, vmin, vmax, vmean, vstd = tensor.shape, tensor.min(), tensor.max(), torch.mean(tensor), torch.std(tensor)
    print('shape:{} | min:{:.3f} | max:{:.3f} | mean:{:.3f} | std:{:.3f}'.format(shape, vmin, vmax, vmean, vstd))


def create_grid(h, w):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_y, grid_x], dim=-1)
    return grid


def create_grid_3d(c, h, w):
    grid_z, grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=c), \
                                            torch.linspace(0, 1, steps=h), \
                                            torch.linspace(0, 1, steps=w)])
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    return grid


class ImageDataset_3D(Dataset):

    def __init__(self, img_path, img_dim):
        '''
        img_dim: new image size [z, h, w]
        '''
        self.img_dim = (img_dim, img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)

        #CT
        # image = np.load(img_path)['data']  # [C, H, W]

        #PAT
        # path = "C:/Users/Admin/Desktop/ex/ex1-rate6-scan15/recon/SPAT-DS/"
        path = "data/rate/prior/"
        files = os.listdir(path)
        list_img_3d = []
        for file in files:
            if "DS_Store" in file:
                continue
            img = Image.open(path + file)
            # img = img.resize((256, 256),Image.ANTIALIAS)
            list_img_3d.append(np.array(img))
        arr_img_3d = np.array(list_img_3d)
        image = arr_img_3d
        print("image shape:{}".format(image.shape))

        # Crop slices in z dim
        center_idx = int(image.shape[0] / 2)
        num_slice = int(self.img_dim[0] / 2)
        image = image[0:5, :, :]
        im_size = image.shape
        print(image.shape, center_idx, num_slice)

        # Complete 3D input image as a squared x-y image
        if not(im_size[1] == im_size[2]):
            zerp_padding = np.zeros([im_size[0], im_size[1], np.int((im_size[1]-im_size[2])/2)])
            image = np.concatenate([zerp_padding, image, zerp_padding], axis=-1)

        # Resize image in x-y plane
        image = torch.tensor(image, dtype=torch.float32)[None, ...]  # [B, C, H, W]
        # image = F.interpolate(image, size=(self.img_dim[1], self.img_dim[2]), mode='bilinear', align_corners=False)

        # Scaling normalization
        image = image / torch.max(image)  # [B, C, H, W], [0, 1]
        self.img = image.permute(1, 2, 3, 0)  # [C, H, W, 1]
        display_tensor_stats(self.img)

    def __getitem__(self, idx):
        grid = create_grid_3d(*self.img_dim) #shengcheng wangge
        return grid, self.img

    def __len__(self):
        return 1



class ImageDataset_2D(Dataset):

    def __init__(self, img_path, img_dim, img_slice):
        '''
        img_dim: new image size [h, w]
        '''
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        # image = np.load(img_path)['data']

        # PAT_regression
        # path = "/public2/zhongyutian/zytNeRP/data/PAT_data/Study_lxpmouse/Scan_2/DS/"
        # files = os.listdir(path)
        # list_img_3d = []
        # for file in files:
        #     if "DS_Store" in file:
        #         continue
        #     img = Image.open(path + file).convert("L")
        #     list_img_3d.append(np.array(img))
        # arr_img_3d = np.array(list_img_3d)

        # PAT_train
        path = "C:/Users/Admin/Desktop/ex/ex2-rate6-scan6/recon/DS/SPAT1/"
        files = os.listdir(path)
        list_img_3d = []
        for file in files:
            if "DS_Store" in file:
                continue
            img = Image.open(path + file).convert("L")
            list_img_3d.append(np.array(img))
        arr_img_3d = np.array(list_img_3d)
        image = arr_img_3d
        print("image shape:{}".format(image.shape))

        image = image[img_slice, :, :]  # Choose one slice as 2D CT image
        imsize = image.shape
        print("image slice shape:{}".format(image.shape))

        # Complete as a squared image
        if not(imsize[0] == imsize[1]):
            zerp_padding = np.zeros([imsize[0], np.int((imsize[0] - imsize[1])/2)])
            image = np.concatenate([zerp_padding, image, zerp_padding], axis=1)

        # Interpolate image to predefined size
        image = cv2.resize(image, self.img_dim[::-1], interpolation=cv2.INTER_LINEAR)

        # Scaling normalization
        image = image / np.max(image)
        self.img = torch.tensor(image, dtype=torch.float32)[:, :, None]
        print("final image shape:{}".format(image.shape)) #2D regression & train : image shape:(92, 256, 256), image slice shape:(256, 256), final image shape:(256, 256)

        display_tensor_stats(self.img)


    def __getitem__(self, idx):
        grid = create_grid(*self.img_dim)
        return grid, self.img

    def __len__(self):
        return 1


class ImageDataset_ISS_2D(Dataset):

    def __init__(self, img_path, img_dim, img_slice):
        '''
        img_dim: new image size [h, w]
        '''
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else tuple(img_dim)
        # image = np.load(img_path)['data']

        # PAT_regression
        # path = "/public2/zhongyutian/zytNeRP/data/PAT_data/Study_lxpmouse/Scan_2/DS/"
        # files = os.listdir(path)
        # list_img_3d = []
        # for file in files:
        #     if "DS_Store" in file:
        #         continue
        #     img = Image.open(path + file).convert("L")
        #     list_img_3d.append(np.array(img))
        # arr_img_3d = np.array(list_img_3d)

        # PAT_train
        path = "/public2/zhongyutian/zytNeRP/data/PAT_data/Study_deadmouse2/WL1/DS/"
        files = os.listdir(path)
        list_img_3d = []
        for file in files:
            if "DS_Store" in file:
                continue
            img = Image.open(path + file).convert("L")
            list_img_3d.append(np.array(img))
        arr_img_3d = np.array(list_img_3d)

        image = arr_img_3d
        print("image shape:{}".format(image.shape))
        image = image[img_slice, :, :]  # Choose one slice as 2D CT image
        imsize = image.shape
        print("image slice shape:{}".format(image.shape))

        # Complete as a squared image
        if not (imsize[0] == imsize[1]):
            zerp_padding = np.zeros([imsize[0], np.int((imsize[0] - imsize[1]) / 2)])
            image = np.concatenate([zerp_padding, image, zerp_padding], axis=1)

        # Interpolate image to predefined size
        image = cv2.resize(image, self.img_dim[::-1], interpolation=cv2.INTER_LINEAR)

        # Scaling normalization
        image = image / np.max(image)
        self.img = torch.tensor(image, dtype=torch.float32)[:, :, None]
        print("final image shape:{}".format(
            image.shape))  # 2D regression & train : image shape:(92, 256, 256), image slice shape:(256, 256), final image shape:(256, 256)

        display_tensor_stats(self.img)

    def __getitem__(self, idx):
        grid = create_grid(*self.img_dim)
        return grid, self.img

    def __len__(self):
        return 1

class ImageDataset(Dataset):

    def __init__(self, img_path, img_dim):
        self.img_dim = (img_dim, img_dim) if type(img_dim) == int else img_dim

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        left_w = int((w - h) / 2)

        image = image[:, left_w:left_w + h]
        image = cv2.resize(image, self.img_dim, interpolation=cv2.INTER_LINEAR)
        self.img = image

    def __getitem__(self, idx):
        image = self.img / 255
        grid = create_grid(*self.img_dim[::-1])

        return grid, torch.tensor(image, dtype=torch.float32)[:, :, None]

    def __len__(self):
        return 1


