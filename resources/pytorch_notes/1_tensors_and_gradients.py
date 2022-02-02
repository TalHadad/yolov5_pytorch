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
