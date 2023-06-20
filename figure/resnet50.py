import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from orgin_utils import GradCAM, show_cam_on_image, center_crop_img
import conformer
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Dataset(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.len=x.shape[0]
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len

data= pd.DataFrame(np.load(r"knock.npy", allow_pickle=True))
data.columns = ['feature', 'label']
x = np.array(data.feature.tolist())
y = np.array(data.label.tolist())
a=torch.tensor(x[0])
b=int(y[0])
train_data=x
train_label=y
train=Dataset(train_data,train_label)

def main():
    model = models.resnet50()
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(2048, 50, bias=True)
    state = torch.load(r'E:\code\voice recongnize\main\resnet50.params')
    model.load_state_dict(state)
    target_layers = [model.layer4[-1]]

    img=a.numpy()
    img1 = torch.squeeze(a).numpy()
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = b# tabby, tabby cat 图片的label
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=torch.unsqueeze(a,0), target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img1, grayscale_cam, use_rgb=True)
    plt.figure(figsize=(6, 4))
    plt.imshow(img.transpose(1, 2, 0))
    plt.figure(figsize=(6, 4))
    plt.xlabel('resnet', fontsize=15)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
