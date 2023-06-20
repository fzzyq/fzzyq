import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from orgin_utils import GradCAM, show_cam_on_image, center_crop_img
from vit_model import vit_base_patch16_224
from torch.utils.data import Dataset
import pandas as pd

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
    model = torchvision.models.vit_b_16(pretrained=True)

    model.conv_proj = torch.nn.Conv2d(1, model.conv_proj.out_channels, kernel_size=(16, 16), stride=(16, 16))
    model.heads = torch.nn.Linear(768, 50, bias=True)
    state = torch.load(r'E:\code\voice recongnize\main\vit.params')
    model.load_state_dict(state)
    # Since the final classification is done on the class token computed in the last attention block,
    # the output will not be affected by the 14x14 channels in the last layer.
    # The gradient of the output with respect to them, will be 0!
    # We should chose any layer before the final attention block.
    target_layers = [model.encoder.layers.encoder_layer_11.ln_1]
    img = a.numpy()
    img1 = torch.squeeze(a).numpy()
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))
    target_category = b  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=torch.unsqueeze(a,0), target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img1, grayscale_cam, use_rgb=True)
    plt.figure(figsize=(6, 4))
    plt.imshow(img.transpose(1, 2, 0))
    plt.figure(figsize=(6, 4))
    plt.xlabel('vit', fontsize=15)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
