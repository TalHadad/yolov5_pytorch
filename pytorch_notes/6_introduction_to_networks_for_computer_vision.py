#############################################################################################
# page 6: Inrtoduction to network for computer vision
#############################################################################################

######################################
# 1. Intro to Convolution
######################################

# 1.1.
# $ Methods: nn.Conv2d
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stribe=2, padding=1)

# 1.2.
# $ Methods: torch.zeros
images = torch.zeros(1, 1, 5, 5) # (1 = number of images, 1 = channel (gray scale = 1), 5,5 = 5x5 = size of image)
images = [0, 0, :, 2] = 1 # color some pixel, 0=black, 1=white
# after applying convolutional on image size MxM the output will be of size ((M-K)/stribe)+1 (K=kernel_size, window_size=KxK)
# after padding image size MxM the output size will be M'=M+2*padding

# 1.3.
# $ Methods: none
conv = nn.Conv2d(in_channels=2, out_channels=3)
images = torch.zeros(1, 2, 5, 5)
# * Figure: see my paper notes for better image/understanding ([2{1]} represent matrix1 over matrix2)
# * Figure cont.: [2{1]}  ->                   [4{3]} }                               = {1}*{3}+[2]*[4] = []
#                 -----                        [6{5]} } 3 sets of kernels,              {1}*{5}+[2]*[6]   []
#         image with 2 channels (can be RGB)   [8{7]} } one for each output channel     {1}*{7}+[2]*[8]   []
#                                             -------
#                                         2 kernels, one for each input channel

# 1.4. manualy setting the weights and bias values.
# $ Methods: conv.state_dict()['weight'], conv.state_dict()['bias']
conv.state_dict()['weight'][0][0] = torch.tensor([[...],...])
conv.state_dict()['bias'][:] = torch.tensor([0,0,...])

# 1.5.
# $ Methods: F.relu
import torch.nn.functional as F
A = F.relu(conv(image))

# 1.6.
# $ Methods: nn.MaxPool2d, stribe=None
max = torch.nn.MaxPool2d(2, stribe=1) # kernel size = 2, stribe size = 1, also stribe = None for no overlaps
new_image = max(image)
# helps with shifting patterns (?)

######################################
# 2. Convolutional Neural Network
######################################

# 2.1.
# $ Methods: nn.Module, nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Linear, flat=out.view(out.size(0),-1)
class CNN(nn.Module):
      def __init__(self, conv_out_1=16, conv_out_2=32):
          super(self, CNN).__init__()
          self.cnn1 = nn.Conv2d(in_channels=1, out_channels=conv_out_1, kernel_size=5, padding=2)
          self.relu1 = nn.ReLU()
          self.maxpool1 = nn.MaxPool2d(kernel_size=2)
          self.cnn2 = nn.Conv2d(in_channels=conv_out1, out_channels=conv_out_2, kernel_size=5, stribe=1, padding=2)
          self.relu2 = nn.ReLU()
          self.maxpool2 = nn.MaxPool2d(kernel_size=2)
          self.fc1 = nn.Linear(conv_out_2*4*4, 10) # the first 4 in conv_out_2*4*4 come from:
                                                   # conv1_size = (((2 * 2=padding + 16=x.size) - 5=kernel_size) / 1=stribe) + 1 = 16
                                                   # after_pool = ((16=conv1_size - 2=max_pool) / 2=max_pool) + 1 = 8
                                                   # conv2_size = (((2 * 2=padding + 8=after_pool) - 5=kernel_size) / 1=stribe) + 1 = 8
                                                   # after_pool = ((8=conv2_size - 2=max_pool) / 2=max_pool) + 1 = 4
     def forward(self, x):
         z1 = self.cnn1(x)
         a1 = self.relu1(z1)
         out = self.maxpool(a1)
         z2 = self.cnn2(out)
         a2 = self.relu2(z2)
         out = self.maxpool2(a2)
         flat_out = out.view(out.size(0), -1)
         z3 = self.fc1(flat_out)
         return z3

# 2.2.
# $ Methods: transforms.Resize, torch.max
for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad() # optimizer = optim.SGD(...)
        z = model(x) # no need to reshape x (x.size=(16,16), composed=transforms.Compose([transforms.Resize((16,16)), transforms.ToTensor())))
        loss = loss_function(z, y) # loss_function=nn.CrossEntropyLoss()
        loss.backward()
        optimizer.step()
    correct = 0
    for x, y in validation_loader:
        z = model(x)
        _, y_hat = torch.max(z.data, 1)
        correct += (y_hat==y).sum().item()
    accuracy = correrct/len(validation_dataset)
    # optional: save accuracy and loss for ploting:
    # accuracy_list.append(accuracy)
    # loss_list.append(loss.data)

######################################
# 3. Pre trained Networks
######################################

# using torchvision, you can create a specific model with:
#       1. random weights
#       2. pretrained weights

# 3.1.
# $ Methods: torchvision.models, models.resnet18, pretrained, transforms.Resize(224), param.requires_grad=False, Linear(512), if param.required_grad
import torchvision.models as models
model = models.resnet18(pretrained=True)

# specific transform for resnet (each model has its own transform)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transform.Compose([transforms.Resize(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)])

train_dataset = dataset(root='./data', download=True, split='test', transform=transform)
validation_dataset = ...

# don't make the model differentiable
for param in model.parameters():
    param.requires_grad=False

model.fc = nn.Linear(512, 3)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([param for param in model.parameters() if param.required_grad], lr=0.001)
# optimize only the parameters that require grad

for epoch ... # same as before
    ... train()
    ... eval()
    # In many models (including ResNet), we must set the model to train() and eval().
    # This can take a long time, even though it's pre-trained because these models are pretty large.
