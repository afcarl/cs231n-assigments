
import numpy as np
import pytest
from numpy.testing import assert_allclose


from cs231n.classifiers.neural_net import TwoLayerNet

# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
  np.random.seed(0)
  return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
  np.random.seed(1)
  X = 10 * np.random.randn(num_inputs, input_size)
  y = np.array([0, 1, 2, 2, 1])
  return X, y

def test_toy_scores():

  net = init_toy_model()
  X, y = init_toy_data()

  scores = net.loss(X)

  correct_scores = np.asarray([
    [-0.81233741, -1.27654624, -0.70335995],
    [-0.17129677, -1.18803311, -0.47310444],
    [-0.51590475, -1.01354314, -0.8504215 ],
    [-0.15419291, -0.48629638, -0.52901952],
    [-0.00618733, -0.12435261, -0.15226949]])

  # The difference should be very small. We get < 1e-7

  assert_allclose(scores, correct_scores, 0.00001)


def test_toy_loss():
  net = init_toy_model()
  X, y = init_toy_data()

  loss, _ = net.loss(X, y, reg=0.1)
  correct_loss = 1.30378789133

  # should be very small, we get < 1e-12
  assert np.sum(np.abs(loss - correct_loss)) < 1e-12
