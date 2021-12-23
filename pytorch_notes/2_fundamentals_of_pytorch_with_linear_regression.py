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
