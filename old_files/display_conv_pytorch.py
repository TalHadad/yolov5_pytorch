import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio
import sys

A = imageio.imread('runs/detect/exp/zidane.jpg')
# Define how the convolution operation works
conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

# from [H, W, C] to [C, H, W]
transposed_image = A.transpose((2, 0, 1))
# add batch dim
transposed_image = np.expand_dims(transposed_image, 0)

image_d = torch.FloatTensor(transposed_image)
fc = conv2(image_d)
fc1 = fc.permute(0, 2, 3, 1)[0]
result = fc1.data.numpy()
max_ = np.max(result)
min_ = np.min(result)
result -= min_
result /= max_

plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(A)
plt.subplot(1,2,2)
plt.imshow(result)

plt.show()