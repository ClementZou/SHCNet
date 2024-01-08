import torch.utils.data as tud
import torch
import numpy as np
import torchvision.transforms as transforms
from einops import rearrange
from option import opt
import scipy

class Dataset(tud.Dataset):
    def __init__(self, file_paths, num_files, bias_list, sim, trans=False, norm=True):
        super(Dataset, self).__init__()
        self.size = opt.size
        self.file_paths = file_paths
        self.num_files = num_files
        self.trans = trans
        self.norm = norm
        self.sim = sim
        self.bias_list = bias_list
        # self.num_files = 2
        '''
        meas = np.zeros((((height, width+bias, frame, mea))))
        mask = np.zeros((((height, width, frame, mea))))
        hsi = np.zeros((((height, width, channel_num, frame))))
        '''
        self.time_ratio = opt.time_ratio
        self.channel = opt.channel

    def __getitem__(self, index):
        data = scipy.io.loadmat(self.file_paths[index])
        hsi = data['gt'][:, :, :, :self.time_ratio]
        mask = data['mask'][:, :, :self.time_ratio]

        if opt.time_ratio == 1:
            mask = mask[:, :, np.newaxis]
            hsi = hsi[:, :, :, np.newaxis]
        if self.trans:
            hsi = hsi.transpose(1, 0, 2, 3)
            mask = mask.transpose()

        if self.sim:
            mea = np.zeros((self.size, self.size + self.bias_list[self.channel - 1]))
            for channel in range(self.channel):
                label_single_channel = hsi[:, :, channel, :].copy() * mask
                mea[:, self.bias_list[channel]:self.bias_list[channel] + self.size] += label_single_channel / self.channel
        else:
            mea = data['mea']
            if self.trans:
                mea = mea.transpose()

        if hsi.dtype == 'uint8':
            divider = 255.0
        elif hsi.dtype == 'uint16':
            divider = 65535.0
        else:
            divider = 1
        hsi = hsi / divider
        hsi[hsi < 0] = 0
        hsi[hsi > 1] = 1
        mea = mea / divider
        mea[mea < 0] = 0
        mea[mea > 1] = 1
        # Normalization
        mea_max = np.max(mea)
        mea_min = np.min(mea)
        transform_mea = transforms.Compose([
            transforms.Lambda(lambda x: (x - mea_min) / (mea_max - mea_min))  # 自定义归一化操作
        ])

        hsi_max = np.max(hsi)
        hsi_min = np.min(hsi)
        transform_hsi = transforms.Compose([
            transforms.Normalize(hsi_min, hsi_max - hsi_min)
        ])

        mask = torch.FloatTensor(mask.copy()).permute(2, 0, 1)
        hsi = torch.FloatTensor(hsi.copy())
        mea = torch.FloatTensor(mea.copy())
        if self.norm:
            hsi = rearrange(hsi, 'h w c t -> (c t) h w')
            hsi = transform_hsi(hsi)
            hsi = rearrange(hsi, '(c t) h w -> c t h w', c=self.channel)
            mea = transform_mea(mea)
        return mea, hsi, mask, self.file_paths[index]

    def __len__(self):
        return self.num_files
