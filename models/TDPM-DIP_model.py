import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import piq
import numpy as np
from diffusers import UNet2DModel

class TDPM_DIP(nn.Module):
    def __init__(self, DDPM, noise_scheduler, truncated_step, device):
        super().__init__()
        self.device = device
        self.DDPM = DDPM.to(self.device)
        self.noise_scheduler = noise_scheduler
        self.truncated_step = truncated_step

    def train_dip(self, x_truncated, noise):

        self.DIP = UNet2DModel(
                 sample_size=64,  # the target image resolution
                 in_channels=3,  # the number of input channels, 3 for RGB images
                 out_channels=3,  # the number of output channels
                 layers_per_block=2,  # how many ResNet layers to use per UNet block
                 block_out_channels=(128, 128, 256, 512),  # the number of output channes for each UNet block
                 down_block_types=(
                    "DownBlock2D",  # a regular ResNet downsampling block
                    "DownBlock2D",
                    "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                    "DownBlock2D",
                ),
                 up_block_types=(
                    "UpBlock2D",  # a regular ResNet upsampling block
                    "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                    "UpBlock2D",
                    "UpBlock2D",
                ),
        )

        self.DIP.to(self.device)
        print("Start Training DIP...")

        self.DIP_iterations = 100
        criterion = nn.MSELoss()
        lr = 1e-4
        optimizer = optim.Adam(self.DIP.parameters(), lr=lr)

        for i in tqdm(range(self.DIP_iterations)):
            out = self.DIP(noise, 0).sample
            loss = criterion(out, x_truncated)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (i+1) % 5 == 0:
            #     print("Loss is {}".format(loss.item()))
            #     plt.subplot(1, 2, 1)
            #     pred_np = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
            #     plt.imshow(pred_np)
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(x_truncated.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5)
            #     plt.show()

    def DIP_denoise(self, x_t):
        x_truncated_pred = self.DIP(x_t, 0).sample
        return x_truncated_pred

    def DDPM_reverse_step(self, x_truncated):
        sample = x_truncated
        for t in tqdm(list(range(0, self.truncated_step+1))[::-1]):
           with torch.no_grad():
             residual = self.DDPM(sample, t).sample
           sample = self.noise_scheduler.step(residual, t, sample).prev_sample

        return sample


    def TDPM_reverse(self, x_t):
        '''
           Step 1.  Use DIP to denoise gaussian noise (x_t) to truncated steps (x_truncated)
                    >>> x_truncated = self.DIP_denoise(self, x_t)

           Step 2.  Use DDPM to execute the reverse steps from x_truncated to x_0
                    >>> x_0 = self.DDPM_reverse_step(x_truncated)

           Step 3.  return x_0
                    >>> return x_0
        '''

        x_truncated = self.DIP_denoise(x_t)
        x_0 = self.DDPM_reverse_step(x_truncated)
        return x_0


    def add_to_truncated(self, noise, image):
        truncated_target = self.noise_scheduler.add_noise(image, noise, torch.LongTensor([self.truncated_step]))
        return truncated_target

    def count_denoising_steps(self):
        return self.DIP_iterations + self.truncated_step

    def forward(self, x_t):
        '''
           Same with self.TDPM_reverse(x_t)
           >>> return self.TDPM_reverse(x_t)
        '''
        return self.TDPM_reverse(x_t)
