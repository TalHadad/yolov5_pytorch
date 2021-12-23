#############################################################################################
# page 0: Introduction
#############################################################################################

######################################
# 1. Introduction
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
