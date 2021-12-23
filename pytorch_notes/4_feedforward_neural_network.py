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
