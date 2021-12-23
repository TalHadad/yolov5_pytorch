#############################################################################################
# page 0: Introduction
#############################################################################################

######################################
# 1. Inrtoduction
######################################

# PyTorch is a replacement for NumPy to use the power of GPUs.
# PyTorch is a deep learning research platform.
# Imperative programing defines computation as you type it (feels more like python)
import torch
a = torch.tensor(1.0)
b = torch.tensor(1.0)
c = a+b # tensor(2.)

# In contrast, Symbolic Programing (such as TensorFlow) that:
# 1. Define the computation
import tensorflow as tf
a = tf.constant(1.0, name='a')
b = tf.constant(1.0, name='b')
c = a+b # <tf.Tensor'add_2:0' shape=() dtype=float32>
# then 2. Execute the computation
sess = tf.Session()
output = sess.ren(c) # tensor(2.)
# is more efficiant, but more difficult to develop new things.

# Pros:
# 1. every computation can be accessed
# 2. easy to debug
# 3. can integrate contral flow statment
# 4. gain insight in yout model
# 5. easier development (than simbolic programing)
# Cons:
# 1. less efficiant

# Conclution:
# TensorFlow is usually in large scale production.
# PyTorch is used for research to test out ideas quickly
# (lower-level environment allows you to experiment with new ideas).
# Keras is simpler to experiment with for standard layers.

#############################################################################################
# page 1: Tensors and gradients
#############################################################################################

######################################
# 1. Tensors
######################################
import torch

# 1.1. One Dimension

# Datatypes:
#   1. torch.float32/64/16 or Float/Double/HalfTensor
#   2. torch.uint8 or ByteTensor
#   3. torch.int8/16/32/64 or Char/Short/Int/LongTensor

# 1.1.1 Types and shape
# $ Methods: .tensor, .dtype, .type, .FloatTensor, .size, .shape, .ndimension, .view
a=torch.tensor([0,1,2,3,4]) # [[row],[row]] for 2D tensor (*)
# (*) Note: The elements in the list that will be converted to tensor must have the same type.
a[0] # tensor(0)
a.dtype # torch.int64
a.type() # torch.LongTensor
a = torch.FloatTensor([0,1,2,3,4]) # create a specific type tensor
a = a.type(torch.LongTensor) # a.type()==torch.LongTensor
a.size() # torch.Size(5, row * columns in 2D tensors
a.shape # (5,), (rows,columns) in 2D tensors
a.ndimension() # 1
a_col = a.view(5, 1) # reshape to (row_num, col_num) (**)
a_col = a.view(-1, 1) # if you don't know the size, use -1 (**)
# (**) Note: The number of elements in a tensor must remain constant after applying view.
#            For tensors with dynamic size use -1 (.view(-1,1)), -1 can represent any size.
#            You can set only one argument as -1.

# 1.1.2. NumPy
# $ Methods: .from_numpy, .numpy
# changing numpy_array will also change torch_tensor and back_to_numpy
# (they point to each other)
# numpy_array <- torch_tensor <- back_to_numpy
numpy_array = np.array([0,1,2,3,4])
torch_tensor = torch.from_numpy(numpy_array)
back_to_numpy = torch_tensor.numpy()

# 1.1.3. Pandas
# $ Methods: .values
pandas_series = pd.Series([0,1,2,3,4])
pandas_to_torch = torch.from_numpy(pandas_seres.values)

# 1.1.4. List
# $ Methods: .tolist
torch_to_list = some_tensor.tolist()

# 1.1.5. Item
# $ Methods: .item
some_tensor[0].item() # 0

# 1.1.6. Indexing and Slicing
# $ Methods: :, ','
some_tensor[0] = 100
some_tensor[1:4] # tensor[1,2,3]
some_tensor[3:5] = torch.tensor([300,400]) # assign to range
some_tensor[3,5] = torch.tensor([300,500]) # assign to specific indexes
some_tensor[3,5] = 100 # assign a single value to seleted indexed in the tensor (you can use only one value for the assignment)

# 1.1.7. Basic operations
# $ Methods: +, *, .dot
tensor_3 = tensor_1+tensor_2 # [1,0]+[0,1] = tensor([1,1])
tensor_4 = 2*tensor_3 # tensor([2,2])
tensor_5 = tensor_1*tensor2 # tensor([1*0,0*1]) = tensor([0,0])
tensor_6 = torch.dot(tensor_1,tensor_2) # 1*0+0*1 = tensor(0)
# np.dot(tensor_1,tensor2) = 0
tensor_7 = tensor_1+1 # (broadcasting) tensor([2,1])

# 1.1.8. Universal Function
# $ Methods: .mean, .max, .std, .sin
mean = some_tensor.mean()
max = some_tensor.max()
standard_deviation = tensor.std()
tensor_sin = torch.sin(some_tensor)

# 1.1.9. Create tensor from start,end,step
# $ Methods: .linspace
torch.linspace(-2,2,step=5)

# 1.1.10 Ploting Mathematical Functions
# $ Methods: .numpy
import matplotlib.pyplot as plt
#%matplotlib inline # in jupyter notebook
plt.plot(x_tensor.numpy(),y_tensor.numpy())

# 1.2. Two Dimension

# 1.2.1. Same operation as one dimension
# $ Methods: .shape, .size, .ndimensions, ',', ':,', ',:', +, *, .mm
a = torch.tensor([0,1],[2,3]]) # [[row],[row]]
a.shape # (2,2), (rows,columns)
a.size() # 4 , rows*columns
a.ndimensions() # 2
a[0][1] # 2
a[0,1] # 2
a[:,1] # [1,3] (***)
# (***) Note: tensor[r1:r2,c1:c2] != tensor [r1:r2][c1:c2]
#             [c1:c2] need new indexes (cn-c1)
# a[0,0:1] # [0,1]
tensor_1+tensor_2 # [1 0] + [2 1] = [1+2 0+1]
                  # [0 1]   [1 2]   [0+1 1+2]
tensor_1*2 # [1*2 0*2]
           # [0*2 1*2]
tensor_1*tensor_2 # (element wise) [1 0] * [2 1] = [1*2 0*1]
                  #                [0 1]   [1 2]   [0*1 1*2]
torch.mm(tensor_1, tensor_2) # [1 2 3] * [7 8] = [1*7+2*9+3*11 1*8+2*10+3*12]
                             # [4 5 6]   [9 10]  [4*7+5*9+6*11 4*8+5*10+6*12]
                             #           [11 12]

######################################
# 2. Derivatives
######################################

# 2.1. Compute derivative
# $ Methods: require_grad=True, .backward, .grad
x = torch.tensor(2., requires_grad=True) # x=2 (tensor(2.,require_grad=2))
y = x**2 # y(x=2) = x^2 = 2^2 = 4 (tensor(4.,grad_fn=<...>))
y.backward() # dy(x=2)/dx = 2x, result = None
x.grad # dy(x=2)/dx = 2*2 = 4 (tensor(4.)), also partial derivative in multivariable function, e.g. y(x,w)=wx+x^2

# 2.2. Must do (?)
# $ Methods: .grad.zero_, retain_graph=True
x.grad.zero_()
y.backward(retain_graph = True)

# 2.3. Detach before numpy
# $ Methods: .detach
# Need to detach every parameter with requires_grad==True before we can cast to numpy array.
x.detach().numpy()
x.grad.detach().numpy()
y.detach().numpy()

# 2.4. Functions
# Methods: .sum, .relu
import torch.nn.functional as F
y = torch.sum(F.relu(x))

# 2.5. Sum for multiple values
# Methods: .sum
y = torch.sum(x**2) # use .sum if x has multiple values, e.g. x=torch.linspace(..., require_grad=True)

######################################
# 3. Dataset Class
######################################

# 3.1.
# $ Methods: Dataset, __getitem__, __len__
from torch.utils.data umport Dataset
class toy_set(Dataset):
      def __init__(self, length=100, transform=None): #dataset=toy_set()
          self.x = 2*torch.ones(length,2)
          self.y = torch.ones(length,1)
          self.length = length
          self.transform = transform # transform class must override __call__ method
      def __getitem__(self, index):
          sample = self.x[index], self.y[index]
          if self.transform:
             sample = self.transform(sample)
          return sample
      def __len__(self): # len(dataset)
          return self.length

# 3.2. Compose transforms
# $ Methods: transforms.Compose
# !conda install -y torchvision (in jupyter notebook)
from torchvision import transforms
data_transform = transforms.Compose([transform_1(), transform_2()])
dataset = toy_set(transform_data_transform) # apply multiple data transforms in __getitem__

# 3.3. Seed
# $ Methods: manual_seed
torch.manual_seed(0) # forces the random function to give same numbers in every run

# 3.4.
# $ Methods: torchvision.datasets, .MNIST
import torchvision.datasets as dsets
dataset = dsets.MNIST(...)
# Prebuilt datasets

# 3.5. Transforms
# $ Methods: .CenterCrop, .ToTensor, .RandomHorizomtalFlip, .RandomVerticalFlip
transforms.Compose([transforms.CenterCrop(20),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomVerticalFlip(p=1)])

#############################################################################################
# page 2: Fundamentals of pytouch with linear regression
#############################################################################################

######################################
# 1. Linear Regression in 1D-prediction
######################################

# 1.1.
# $ Methods: .nn, Linear
from torch.nn import Linear
model = Linear(in_features=1, out_features=1)
y = model(x)

# 1.2.
# $ Methods: .manual_seed
torch.manual_seed(1) # the parameters are randomly selected to be the same in each run

# 1.3.
# $ Methods: .parameters
list(model.parameters()) # to see the values of the model parameters, list() due to lazy iterator

# 1.4. Custom Module
# $ Methods: .Module, super, __init__, forward
import torch.nn as nn
class LR(nn.Module):
    def __init__(self, in_size, out_size):
        super(LR, self).__init__() # or nn.Module.__init__(self)
        self.linear = nn.Linear(in_size, out_size)
    def forward(self, x):
        out = self.linear(x)
        return out

######################################
# 2. Linear Regression Training
######################################

# 2.1. Training Parameters in PyTorch |The Hard Way|
# $ Methods: .arange, .randn, epoch, .data, .grad.data, .grad.data.zero_
x = torch.arange(-3, 3, 0.1).view(-1,1)
w = torch.tensor(-10.0, requires.grad=True)
y = -3*x+torch.randn(x.size())
def forward(x):
    return w*x
def loss(y_hat, y):
    return torch.mean((y_hat-y)**2) # MSE
LOSS = [] # to plot the loss
for epoch in range(epochs): # epochs=3
    loss = loss(forward(x),y)
    loss.backward()
    w.data = w.data-lr*w.grad.data # lr=0.1
    w.grad.data.zero_()
    LOSS.append(loss) # to plot the loss

# 2.2. Learning and updating weights and bias
# $ Methods: none
# delta_l(w,b) = [partial_dl(w,b)/partial_dw]
#                [partial_dl(w,b)/partial_db]
# b = torch.tensor(0., requires_frad=True)
# def forward(x):
#     return w*x+b
for epoch in range(epochs):
    y_hat = forward(x)
    loss = loss(y_hat, y)
    loss.backward()
    w.data = w.data - lr*w.grad.data
    w.grad.data.zero_()
    b.data = b.data - lr*b.grad.data
    b.grad.data.zero_()

######################################
# 3. Stochastic, Batch and Mini-Batch Gradient Descent
######################################
# Stochastic Gradient Descent:
# for each epoch:
#    for each batch (single randomly selected instance):
#       feed forward and backwards
#       and update weights and bias using the batch
# The problem with stochastic gradient descent is that
# our loss curve will exhibit large jumpls for outliner points
# (samples that doesn't really fit in with the data).

# 3.1. Stochastic Gradient Descent
# $ Methods: epoch, x, y, DataLoader
for epoch in range(epochs):
    for x, y in zip(x_train, y_train): # or DataLoader(dataset=toy_set, batch_size=1) insead of zip, class toy_set(Datadet) (from torch.utils.data import Dataset,DataLoader)
        y_hat = forward(x)
        loss = loss(y_hat, y)
        loss.backward()

        w.data = w.data - lr.w.grad.data
        w.grad.data.zero_()

        b.data = b.data - lr.b.grad.data
        b.grad.data.zero_()

# 3.2. Mini-Batch Gradient Decent
# $ Methods: batch_size, trainloader
train_loader = DataLoader(dataset=toy_set, batch_size=4) # batches of 4, iteration=(training_size/batch_size)
# training:
# for epoch ...
#     for x, y in train_loader

# 1. Batch Gradient Descent: for epoch in epochs: forward, backward, update
# 2. Stochastic Gradient Descent: for epoch in epochs: for x, y in batch_of_1: forward, backward, update
# 3. Mini-Batch Gradient Descent: for epoch in epochs: for x, y in batch_of_k: forward, backward, update
# Simpler explanation:
# 1. Batch Gradient Descent: use all of the data/batch (slow in big datasets)
# 2. Stochastic Gradient Descent: use randomly selected instance (fast in big datasets, but loss jumps up and down, final result is good, not best)
# 3. Mini-Batch Gradient Descent: use randomly selected set of instances (fast in big datasets, loss more stable) (the middle between the batch and stochastic)

######################################
# 4. PyTorch Way
######################################
from torch import nn, optim

# 4.1.
# $ Methods: .MSELoss
loss_function = nn.MSELoss()

# 4.2.
# $ Methods: optim, optim.SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4.3.
# $ Methods: .zero_grad, .step
for epoch in range(epochs):
    for x, y in trainloader:
        h_hat = model(x)
        loss = loss_function(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # taking care for us of: w.data = w.data - lr * w.grad.data
                         #                        b.data = b.data - lr * b.grad.data

# 4.4.
# $ Methods: .Linear
model = nn.Linear(input_size, output_size)

# 4.5. optional - initialise the parameters manualy (pytorch already do it for us automaticaly)
# $ Methods: none
model.state_dict()['linear.weight'][0] = -15
model.state_dict()['linear.bias'] = -10

######################################
# 5. Model Validation
######################################
# Parameters: weights, bias
# Hyperparameters: mini-batch, size, learning rate

# 5.1.
# $ Methods: experiment, learning_rates, model, optimizeet, epoch, x, y, train_loss, train_error, validation_error
learning_rates = [0.0001, 0.001, 0.01, 0.1]
for experiment, lr  in enumerate(learning_rates):
    model = linear_regression(1,1)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for x,y in trainloader: # trainloader = DataLoader(dataset=train_dataset, batch_size=1)
                                # train_dataset = Toy_dataset(train=True)
            y_hat = model(x)
            loss = loss_function(y_hat, y) # loss_function = nn.MSELoss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_y_hat = model(train_dataset.x)
    train_loss = loss_function(train_y_hat, train_dataset.y)
    train_error[i] = train_loss.item() # also cost of total loss

    # validation
    validation_y_hat = model(validatin_dataset.x)
    validation_loss = loss_function(validation_y_hat, validation_dataset.y)
    validation_error[i] = validation_loss.item # cost or total loss terms apply only when training?
    MODELS.append(model)

######################################
# 6. Early Stoping
######################################

# 6.1.
# $ Methods: .save, .state_dict, .load_state_dict, .load
torch.save(model.state_dict(), 'best_model.pt')
best_model = linear_regression(1,1) # class linear_regression(nn.Module):
best_model.load_state_dict(torch.load('best_model.pt'))

# 6.2.
# $ Methods: if validation_error < min, .save
min_loss = 1000
for epoch in range(epochs):
    for x,y in trainloader:
        ...
        train_error = ... # LOSS_TRAIN.append(train_loss)
        validation_error = ... # LOSS_VALIDATION.append(validation_loss)
        if validation_error < min_loss:
           min_loss = validation_error
           torch.save(model.state_dict(), 'best_model.pt')

######################################
# 7. Higher Dimensianal Linear Regression
######################################

# 7.1.
# $ Methods: in, out
model = nn.Linear(in_features=3, out_features=2) # (y) [ , ]     =(x) [x1,x2,x3] * (w) [w1, ]     + (b) [ , ]
                                                 #     [...]          [  ...   ]       [w2, ]
                                                 #                                     [w3, ]
                                                 # (weight=out=2)=(w=in=3)       *(w=out=2,h=in=3)+(w=out=2)
x = torch.tensor([[1,2,3]])
y_hat = model(x) # y = [x1w1+x2w2+x3w3+b, x1w4+x2w5+x3w6+b2]
# y = tensor([[x1w1+x2w2+x3w3+b, x1w4+x2w5+x3w6+b2]])

#############################################################################################
# page 3: Logistic and softmax regression
#############################################################################################

######################################
# 1. Logistic Regression Prediction
######################################
import torch.nn as nn

# 1.1.
# $ Methods: .Sigmoid
sig = nn.Sigmoid()
y_hat = sig(z)

# 1.2.
# $ Methods: .sigmoid
import torch.nn.functional as F # or from torch import sigmoid (?)
y_hat = F.sigmoid(z) # actual function

# 1.3.
# $ Methods: .Sequential
model = nn.Sequential(nn.Linear(1,1), nn.Sigmoid()) # sigmoid(xw+b)
y_hat = model(x)

# 1.4. Custom Module
# $ Methods: none
import torch.nn as nn
class logistic_regression(nn.Module):
      def __init__(self, in_size, out_size):
          super(logistic_regression, self).__init__()
          self.linear = nn.Linear(in_size, out_size)
      def forward(self,x):
          return F.sigmoid(self.linear(x))

######################################
# 2. Training Logistic Regression
######################################
# Logistic Regression: P(Y|tetha,x) = mul{n=1,N}((sigmoid(w*x_n+b)^y_n)*(1-sigmoid(w*x_n+b)^(1-y_n)))
# max(p) = max(log(p)) = min(-log(p)) => min(-(1/N)log(p))
# Cross Entropy: l(w) = -(1/N)sum{n=1,N}(y_n*ln(sigmoid(w*x_n+b))+(1+y_n)*ln(1-sigmoid(w*x_n+b)))

# 2.1.
# $ Methods: .mean, .log, .log
def loss_function(y_hat,y):
    return -1*torch.mean(y*torch.log(y_hat)+(1-y)*torch.log(1-y_hat))

# 2.2.
# $ Methods: .BCELoss
cross_entropy_loss_function = nn.BCELoss() # same as 2.1.

# 2.3. # accuracy
# $ Methods: .ByteTensor
y_hat = model(data_set.x) # model is an instance of nn.Module
lable = y_hat > 0.5
print('accuracy=', torch.mean((lable==data_set.y.type(torch.ByteTensor)).type((torch.float)))) # mean of (1 if y_hat==y and y_hat>0.5)

######################################
# 3. Softmax Regression
######################################

# 3.1.
# $ Methods: .max
z = model(x) # model of class nn.Module, Linear(2,3)
_, y_hat = z.max(1) # return the column index with the maximum value,
# e.g. z = [2,*10*,5] -> z.max = tensor([1,1,2])
#          [3,*8*, 5]            (columns 1,1,2)
#          [0, 2,*5*]
# _ is the actual values [10,8,5]
# y_hat is the indexes/classes with highest probability

# 3.2.
# $ Methods: .CrossEntropyLoss
loss_function = nn.CrossEntropyLoss()

# 3.3.
# $ Methods: .max
model = nn.Sequential(nn.Linear(1,3))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
trainloader = DataLoader(dataset = dataset, batch_size = 5) # dataset class is torch.utils.data Dataset
for epoch in range(epochs): # epochs = 300
    for x, y in trainloader:
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()

# accuracy
z = model(dataset.x)
_, y_hat = z.max(1)
correct = (dataset.y == y_hat).sum().item()
accuracy = correct/len(datasey)
print('accuracy = ', accuracy)

# 3.4. probabilities
# $ Methods: dim, .max
softmax_fn = nn.Softmax(dim = -1)
probabilities = softmax_fn(z)
print('probability of class ', torch.max(probabilities).item())

# 3.5.
# $ Methods: .view, .max
for epoch ...
    for x, y in trainloader
        ...
    # accuracy in batches
    correct = 0
    for x_validation, y_validation in validationloader:
        z = model(x_validation.view(-1, 28*28) # flatten the image
        _, y_hat = torch.max(z.data, 1)
        correct += (y_hat == y_validation).sum().item()
    accuracy = correct/len(validation_data)

#############################################################################################
# page 4: Feedforward neural network
#############################################################################################

######################################
# 1. Neural Networks
######################################

# 1.1.
# $ Methods: nn.Module, nn.Linear, F.sigmoid
class Net(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_size)
    def forward(self, x):
        hidden_out = F.sigmoid(self.linear1(x))
        out = F.sigmoid(self.linear2(hidden_out))
        return out

# 1.2. another way to build neural netwoek with sequential module (same as 1.1.)
# $ Methods: nn.Sequential, nn.Linear, nn.Sigmoid
model = torch.nn.Sequential(torch.nn.Linear(in_size, hidden_size),
                            torch.nn.Sigmoid(),
                            torch.nn.Linear(hidden_size, out_size),
                            torch.nn.Sigmoid())

# 1.3. because our output is a logistic unit (why not Maximun Likelihood Estimator?)
# $ Methods: nn.BCELoss
loss_function = nn.BCELoss()

# 1.4. Mean Square Error loss (also for logistic regresion?) (for linear regression is ordinary least squares)
# $ Methods: nn.MSELoss
loss_function = nn.MSELoss()

# Note: 1. y:torch.LongTensor
#       2. y:N (not Nx1 as in logistic regression)

# 1.5. For using neural network for regression, simply remove the sigmoid from the last layer (in 1.1.)
# $ Methods: none
out = (self.linear2(hidden_out))

######################################
# 2. Back Propagation
######################################

# didn't see in my notes

######################################
# 3. Activation Functions
######################################

# 1. Sigmoid: start at 0, end with 1 -> derivative = *y=bell/normal distribution diagram with x=0 (? not 0.5) at the middle* -> vanishing gradient
# 2. Tanh: start at -1, end with 1 -> derivative = *y=bell/normal distribution diagram with x=0 at the middle* -> vanishing gradient
# 3. Relu: start at 0 end with inf -> derivative = *y=flat 0 line up to x=0, and flat 1 line from x=0* -> no vanishing gradient

# 3.1.
# $ Methods: F.tanh
hidden_out = F.tanh(self.linear1(x))

# 3.2.
# $ Methods: F.relu
hidden_out = F.relu(self.linear1(x))

# 3.3.
# $ Methods: nn.Sequential, nn.Linear, nn.Tanh
model = nn.Sequential(nn.Linear(in_size, hidden_size),
                            nn.Tanh(),
                            nn.Linear(hidden_size,out_size))

# 3.4.
# $ Methods: nn.Sequential, nn.Linear, nn.ReLU
model = nn.Sequential(nn.Linear(in_size, hidden_size),
                      nn.ReLU(),
                      nn.Linear(hidden_size, out_size))

######################################
# 4. Build Deep Networks in Pytorch
######################################

# 4.1. using nn.Module
# $ Methods: nn.Module, nn.Linear, F.sigmoid
class Net(nn.Module):
      def __init__(self, in_size, hidden1_size, hidden2_size, out_size):
          super(Net, self).__init__()
          self.linear1 = nn.Linear(in_size, hidden1_size)
          self.linear2 = nn.Linear(hidden1_size, hidden2_size)
          self.linear3 = nn.Linear(hidden2_size, out_size)
      def forward(self, x):
          hidden1_out = F.sigmoid(self.linear1(x))
          hidden2_out = F.sigmoid(self.linear2(hidden1_out))
          out = self.linear3(hidden2_out)
          return out

# 4.2. using nn.Sequential
# $ Methods: nn.Sequential, nn.Linear, nn.Sigmoid
model = nn.Sequential(nn.Linear(in_size, hidden1_size),
                      nn.Sigmoid(),
                      nn.Linear(hidden1_size, hidden2_size),
                      nn.Sigmoid(),
                      nn.Linear(hidden_2, out_size))

# 4.3. using nn.ModuleList
# $ Methods: nn.Module, nn.ModuleList, .append,  nn.Linear, F.relu
class Net(nn.Module):
      def __init__(self, layers):
          super(Net, self).__init__()
          self.hiddens = nn.ModuleList()
          for in_size, out_size in zip(layers, layers[1:]):
              self.hiddens.append(nn.Linear(in_size, out_size))
      def forward(self, x):
          for i, hidden in enumerate(self.hiddens):
              if i<len(self.hidden)-1:
                 x = F.relu(hidden(x))
              else
                out = hidden(x)
          return out

#############################################################################################
# page 5: Deep networks
#############################################################################################

######################################
# 1. Dropout
######################################

# Each neuron is turned off (or 'killed', multiplied by zero) with probability p.
# PyTorch normalize the fvalues in the training phase 1/(1-p) (p in [0,1] -> 1/(1-p) in [1,inf)) (e.g. p=0.2).
# p=0 don't kill any neurons, p=1 kill all neurons.
# In PyTorch you can select different p for each layer, usually the more neurons in the layer higher probability.

from torch import nn, F

# 1.1. using nn.Module
# $ Methods: nn.Dropout, self.drop, F.relu, self.linear1
class Net(nn.Module):
      def __init__(self, layers, p=0):
          ...
          self.drop = nn.Dropout(p=p)
          ...
      def forward(self, x):
          ...
          x = self.drop(x) # or x = F.relu(self.drop(self.linear1(x)))
          ...

# 1.2. using nn.Sequential
# $ Methods: nn.Drop
model = nn.Sequential(..., nn.Dropout(0.5), ...)

# 1.3. must set your model to train when using dropout
# $ Methods: model.train
model.train()
for epoch ...
    for x, y ...

# 1.4. must set your model to evaluate before the prediction step
# $ Methods: model.eval
model.eval()
y_hat = model(x)

# 1.5.
# $ Methods: optim.Adam, model.parameters, nn.CrossEntropyLoss, model.eval, model.train
model = Net(2, 300, 2, p=0.5)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
for each ...
   for x, y ...
   TRAIN_LOSS.appeand(loss.item())
   model.eval()
   VAL_LOSS.appeand(loss_function(model(val_dataset.x), val_dataset.y).item())
   model.train()

######################################
# 2. Initializatin
######################################

# 2.1. Random: uniform between [-1, 1] (w in [-1, 1]) *Figure: 2-D graph where w is x and p is y, flat line of p=0 until w=-1, flat line of p=0.5 from w=-1 until w=1, flat line of p=0 from w=1*
# $ Methods: weight.data.uniform_
linear = nn.Linear(in_size, out_size)
linear.weight.data.uniform_

# 2.2. PyTorch default: uniform between [-1/sqr(hidden_size), 1/sqr(hidden_size)] *Figure: hidden_size{ [column_representing_layer] w =>(arrows_to_another_layer) [another_layer]*
# $ Methods: none

# 2.3. Xavier (for the tanh function): uniform between [-sqr(6)/sqr(hidden_out_size+hidden_in_size), sqr(6)/sqr(hidden_out_size+hidden_in_size)] *Figure: hidden_in_size{ [column_representing_layer] w =>(arrows_to_another_layer) [another_layer] w => hidden_out_size{ [another_layer]*
# $ Methods: nn.init.xavier_uniform_
linear = nn.Linear(in_size, out_size)
nn.init.xavier_uniform_(linear.weight)

# 2.4. He (for the relu function):
# $ Methods: nn.init.kaiming_uniform_
linear = nn.Linear(in_size, out_size)
nn.init.kaiming_uniform_(linear.weight, nonlinear='relu')

# 2.5.
# $ Methods: data.uniform_, nn.init.xavier_uniform_, nn.init.kaiming_uniform_, F.relu, F.tanh
class Net(nn.Module):
      def __init__(self, layers): # layers = [in_size, h1_size, h2_size, ..., out_size]
          super(Net, self).__init__()
          self.hiddens = nn.ModuleList()
          for in_size, out_size in zip(layers, layers[1:]):
              linear = nn.Linear(in_size, out_size)
              # as in 2.1.
              linear.weight.data.uniform_(0, 1) # optional, if not mentioned then the default is 2.2.
              # or as in 2.3.
              # nn.init.xavier_uniform_(linear.weight)
              # or as in 2.4.
              # nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
              self.hiddens.append(linear)
      def forward(self, x):
          for i, layer in enumerate(self.hiddens):
              if i<len(self.hiddens)-1:
                 # for 2.1., 2.2., 2.4.
                 x = F.relu(layer(x))
                 # or for 2.1., 2.2., 2.3.
                 # x = F.tanh(layer(x))
              else:
                out = layer(x)
          return out

######################################
# 3. Gradient Descent with Momentum
######################################

# 3.1.
# $ Methods: optim.SGD, model.parameters, momentum
optimizer = torch.optim.SGD(model.parameters, lr=0.1, momentum=0.4)

######################################
# 4. Batch Normalization
######################################

# (this explaination is better in my paper notes)
# Mini-Batch (size M) { z_1 -> ^z_1=(z_1-u_B,1)/sqr(pow(sigma_B,1)+epsilon) -> ~z_1 = gamma_1*^z_1+beta_1 -> a_1(~z_1)
#                     { ...
#                     { z_i -> (normalizaion) ^z_i=(z_i-u_B,i)/sqr(pow(sigma_B,i)+epsilon) -> (scale and shift) ~z_i = gamma_i*^z_i+beta_i -> (instead of zi) a_i(~z_i)
#                     { ...
#                     { z_M -> ...
# u_B,i = (1/M)*sum{m=1,M}(z_i,m) (sum of all z_i values from all the batch).
# pow(sigma_B,i) = (1/M)*sum{m=1,M}pow(z_i,m-u_B,i) (z_i,m => i is neuron num, m is sample num).
# gamma_i, beta_i are learned by optimization.

# For evaluation we use E(z_i)=u_i, var(z_i)=pow(sigma_i) of the validation dataset.

# 4.1. Build model with nn.BatchNorm1()
# $ Methods: nn.BatchNorm1d, F.sigmoid
class Net(nn.Module):
      def __init__(self, in_size, h1_size, h2_size, out_size):
          super(Net, self).__init__()
          self.linear1 = nn.Linear(in_size, h1_size)
          self.linear2 = nn.Linear(h1_size, h2_size)
          self.linear3 = nn.Linear(h2_size, out_size)
          self.bn1 = nn.BatchNorm1d(h1_size)
          self.bn2 = nn.BatchNorm1d(h2_size)
      def forward(self, x):
          hidden1_out = F.sigmoid(self.bn1(self.linear1(x)))
          hidden2_out = F.sigmoid(self.bn2(self.linear2(hidden1_out)))
          out = self.linear3(hidden2_out)
          return out

# 4.2. must train when using batch normalization
# $ Methods: model.train
model.train()
for epoch ...

# 4.3. must eval when using batch normalization
# $ Methods: model.eval
model.eval()
y_hat = model(x)

# Note: when using Batch Normalization:
#       1. no need for Dropout
#       2. no need for Bias
#       3. you can increase learning rate
# Batch normalization reduce Interanal Covariate Shift:
# *Figure: elipse boul in 2-D (like elipse dart board) (0,0) in the middle, and the gradient/arrows is zig-zag from outsize to the center and beyond* -> Batch Normalization -> *Figure: round boul in 2-D (like dart board) (0,0) in the middle, and the gradient/arrows is going straight to the middle*

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
