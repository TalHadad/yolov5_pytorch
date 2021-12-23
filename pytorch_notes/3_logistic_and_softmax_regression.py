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
