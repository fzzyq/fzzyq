from librosa import feature
import librosa.display
import pandas as pd
import tqdm
import numpy as np
import torch
from augment import SpecAugment
import d2l
from d2l import torch as d2l
import matplotlib.pyplot as plt
import random


def noisy(data, noise_factor=3):  #插入随机噪音
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(augmented_data, x_axis='time', y_axis='mel',
                             fmax=8000)
    augmented_data=torch.tensor(augmented_data)
    augmented_data= (augmented_data - augmented_data.mean()) / augmented_data.var()
    augmented_data=torch.unsqueeze(augmented_data,dim=0)
    return augmented_data

def zzf(mel_spectrogram,type=None):
    apply = SpecAugment(mel_spectrogram, type)

    time_warped = apply.time_warp()
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(time_warped[0, :, :, 0].numpy(), x_axis='time',
                             y_axis='mel', fmax=8000)
    e1 = time_warped.numpy()
    e1 = torch.tensor(e1)
    e1 = torch.squeeze(e1)
    e1 = torch.unsqueeze(torch.tensor(e1), dim=0)
    e1 = (e1 - e1.mean()) / e1.var()

    freq_masked = apply.freq_mask()

    e2 = freq_masked.numpy()
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(freq_masked[0, :, :, 0].numpy(), x_axis='time', y_axis='mel',
                             fmax=8000)
    e2 = torch.tensor(e2)
    e2 = torch.squeeze(e2)
    e2 = torch.unsqueeze(torch.tensor(e2), dim=0)
    e2 = (e2 - e2.mean()) / e2.var()

    time_masked = apply.time_mask()
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(time_masked[0, :, :, 0].numpy(), x_axis='time', y_axis='mel',
                             fmax=8000)
    e3 = time_masked.numpy()
    e3 = torch.tensor(e3)
    e3 = torch.squeeze(e3)
    e3 = torch.unsqueeze(torch.tensor(e3), dim=0)
    e3 = (e3 - e3.mean()) / e3.var()

    return e1,e2,e3

data = pd.read_csv(r'E:\code\voice recongnize\ESC-50-master\meta/esc50.csv') #音频文件名，以csv文件保存

valid_data = data[['filename','classID']]
dataset=[]
tt=['LB','LD','SM','SS']
i=0
#测试数据集
for row in tqdm.tqdm(valid_data.itertuples()):
        ff='E:\code/voice recongnize\ESC-50-master/audio10/' +f'{row.filename}'  #音频数据路径
        y1, sr1 = librosa.load(ff, duration=5.18)
        mel_spectrogram= feature.melspectrogram(y=y1,n_mels=224,hop_length=512,n_fft=1024)
        mel_spectrogram= librosa.power_to_db(mel_spectrogram)
        plt.figure(figsize=(6, 4))
        librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel',
                                 fmax=8000)

        c=torch.unsqueeze(torch.tensor(mel_spectrogram), dim=0)
        # if c.shape != (1,224,224):
        #     print(f'{row.filename}'+'_'+f'{row.classID}')
        #     continue
        c=(c-c.mean())/c.var()
        noise = noisy(mel_spectrogram)

        # autotune = tune(mel_spectrogram, sr1)
        # i+=1
        e1, e2, e3 = zzf(mel_spectrogram, type='LB')
        d2l.plt.show()


