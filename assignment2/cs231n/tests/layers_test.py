import numpy as np
import pytest
from numpy.testing import assert_allclose
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

def assert_close(x, y, delta=1e-7):
  assert np.sum(np.abs(x - y)) < delta

# Test the affine_forward function
def test_affine_forward():
  num_inputs = 2
  input_shape = (4, 5, 6)
  output_dim = 3

  input_size = num_inputs * np.prod(input_shape)
  weight_size = output_dim * np.prod(input_shape)

  x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
  w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
  b = np.linspace(-0.3, 0.1, num=output_dim)

  out, _ = affine_forward(x, w, b)
  correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                          [ 3.25553199,  3.5141327,   3.77273342]])

  # Compare your output with ours. The error should be around 1e-9.
  print 'Testing affine_forward function:'
  assert correct_out.shape == out.shape
  assert_close(out, correct_out)


def test_affine_backwards():
  # Test the affine_backward function

  x = np.random.randn(10, 2, 3)
  w = np.random.randn(6, 5)
  b = np.random.randn(5)
  dout = np.random.randn(10, 5)

  dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
  dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
  db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

  _, cache = affine_forward(x, w, b)
  dx, dw, db = affine_backward(dout, cache)
  assert_close(dx, dx_num)
  assert_close(dw, dw_num)
  assert_close(db, db_num)
  # The error should be around 1e-10
  print 'Testing affine_backward function:'
  print db.shape, db_num.shape


def test_relu_forward():
  # Test the relu_forward function

  x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

  out, _ = relu_forward(x)
  correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                          [ 0.,          0.,          0.04545455,  0.13636364,],
                          [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

  # Compare your output with ours. The error should be around 1e-8
  print 'Testing relu_forward function:'
  print 'difference: ', assert_close(out, correct_out)

def test_relu_backwards():
  x = np.random.randn(10, 10)
  dout = np.random.randn(*x.shape)

  dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

  _, cache = relu_forward(x)
  dx = relu_backward(dout, cache)

  # The error should be around 1e-12
  print 'Testing relu_backward function:'
  print 'dx error: ', assert_close(dx_num, dx)

def test_batch_norm():
  # Check the training-time forward pass by checking means and variances
  # of features both before and after batch normalization

  # Simulate the forward pass for a two-layer network
  N, D1, D2, D3 = 200, 50, 60, 3
  X = np.random.randn(N, D1)
  W1 = np.random.randn(D1, D2)
  W2 = np.random.randn(D2, D3)
  a = np.maximum(0, X.dot(W1)).dot(W2)

  print 'Before batch normalization:'
  print '  means: ', a.mean(axis=0)
  print '  stds: ', a.std(axis=0)

  # Means should be close to zero and stds close to one
  print 'After batch normalization (gamma=1, beta=0)'
  a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})
  print "mean"
  assert_close(a_norm.mean(axis=0), 0.0)

  print 'std',
  assert_close(a_norm.std(axis=0), 1.0)

  # Now means should be close to beta and stds close to gamma
  gamma = np.asarray([1.0, 2.0, 3.0])
  beta = np.asarray([11.0, 12.0, 13.0])
  a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
  print 'After batch normalization (nontrivial gamma, beta)'
  print '  means: ', a_norm.mean(axis=0)
  print '  stds: ', a_norm.std(axis=0)
