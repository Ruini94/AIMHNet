from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from glob import glob
from torchvision.transforms.functional import to_tensor
import random
import torchvision

p = np.random.choice([0, 1])  # 这样就是成对翻转的
hori_flip = torchvision.transforms.RandomHorizontalFlip(p)
class MyDataset(Dataset):
    def __init__(self, root, name, cropSize, mode='train'):
        self.root = root
        self.name = name
        self.mode = mode
        self.cropSize = cropSize

        if self.name == "Rain800" or self.name == "Rain1200":
            self.files = sorted(glob(os.path.join(self.root, self.mode, "*.*")))

        else:
            self.files_A = sorted(glob(os.path.join(self.root, self.mode + "A", "*.*")))
            self.files_B = sorted(glob(os.path.join(self.root, self.mode + "B", "*.*")))
            self.files = self.files_A

    def __getitem__(self, index):
        # 在 getitem 里面裁剪一个图成两半
        # 在 getitem 中永远是用 PIL.Image.open 打开文件的
        if self.name == "Rain200L" or self.name == "Rain200H" or self.name == "Raindrop":
            imgA = np.array(Image.open(self.files_A[index]))
            imgB = np.array(Image.open(self.files_B[index]))

        elif self.name == "Rain800" or self.name == "Rain1200":
            img = Image.open(self.files[index])
            w, h = img.size[0], img.size[1]
            imgA = np.array(img.crop((0, 0, w / 2, h)))
            imgB = np.array(img.crop((w / 2, 0, w, h)))

        elif self.name == "RainCityscapes":
            imgA = np.array(Image.open(self.files_A[index]))
            imgB = np.array(Image.open(self.files_B[index // 36]))

        imgA = to_tensor(imgA)
        imgB = to_tensor(imgB)

        hh, ww = imgA.shape[1], imgB.shape[2]

        if self.mode == "train":
            rr = random.randint(0, hh - self.cropSize)
            cc = random.randint(0, ww - self.cropSize)
            imgA = hori_flip(imgA[:, rr:rr + self.cropSize, cc:cc + self.cropSize])
            imgB = hori_flip(imgB[:, rr:rr + self.cropSize, cc:cc + self.cropSize])

            if np.random.random() > 0.5:
                cut_ratio = np.random.rand() / 2
                ch, cw = np.int(self.cropSize * cut_ratio), np.int(self.cropSize * cut_ratio)
                cy = np.random.randint(0, self.cropSize - ch + 1)
                cx = np.random.randint(0, self.cropSize - cw + 1)
                imgA[:, cy:cy + ch, cx:cx + cw] = imgB[:, cy:cy + ch, cx:cx + cw]

        else:
            if hh % 8 != 0 and ww % 8 != 0:
                imgA = imgA[:, :hh-1, :ww-1]
                imgB = imgB[:, :hh-1, :ww-1]
            elif hh % 8 == 0 and ww % 8 != 0:
                imgA = imgA[:, :, :ww-1]
                imgB = imgB[:, :, :ww-1]
            elif hh % 8 != 0 and ww % 8 == 0:
                imgA = imgA[:, :hh-1, :]                    
                imgB = imgB[:, :hh-1, :]                    

        return imgA, imgB

    def __len__(self):
        return len(self.files)
