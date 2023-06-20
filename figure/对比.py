import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils_cnn import GradCAM, show_cam_on_image, center_crop_img
import utils_tf
import conformer
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ReshapeTransform:
    def __init__(self, model):
        input_size = (224,224)
        patch_size = (16,16)
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result

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
    model = conformer.Conformer(in_chans=1,embed_dim=96,num_heads=6,num_classes=50,base_channel=16)
    state = torch.load(r'E:\code\voice recongnize\conformer-main\voice.params')
    model.load_state_dict(state)
    target_layers = [model.conv_trans_12.fusion_block]
    target_layers_tf = [model.conv_trans_12.trans_block.norm1]

    img=a.numpy()
    img1=torch.squeeze(a).numpy()
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    cam_tf = utils_tf.GradCAM(model=model,
                  target_layers=target_layers_tf,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))

    target_category = b# tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=torch.unsqueeze(a,0), target_category=target_category)
    grayscale_cam_tf = cam_tf(input_tensor=torch.unsqueeze(a, 0), target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img1,grayscale_cam,use_rgb=True)
    grayscale_cam_tf= grayscale_cam_tf[0, :]
    visualization_tf = show_cam_on_image(img1, grayscale_cam_tf, use_rgb=True)

    plt.figure(figsize=(6, 4))
    plt.imshow(img.transpose(1, 2, 0))
    # plt.figure(figsize=(6, 4))
    # plt.xlabel('conformer_CNN', fontsize=15)
    # plt.imshow(visualization)
    plt.figure(figsize=(6, 4))
    plt.xlabel('conformer_transformer', fontsize=15)
    plt.imshow(visualization_tf)
    plt.show()


if __name__ == '__main__':
    main()
