import numpy as np
import pytest
from numpy.testing import assert_allclose
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver


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
  print 'difference: ', rel_error(out, correct_out)
  assert np.sum(np.abs(out - correct_out)) < 1e-12
