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
