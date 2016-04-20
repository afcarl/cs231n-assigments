
import numpy as np
import pytest
from numpy.testing import assert_allclose
from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.gradient_check import eval_numerical_gradient

# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

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

def test_toy_gradient():
  # Use numeric gradient checking to check your implementation of the backward pass.
  # If your implementation is correct, the difference between the numeric and
  # analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

  net = init_toy_model()
  X, y = init_toy_data()

  loss, grads = net.loss(X, y, reg=0.1)

  # these should all be less than 1e-8 or so
  for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.1)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)

    param_grad = grads[param_name]
    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, param_grad))

    # pytest.set_trace()
    assert rel_error(param_grad_num, grads[param_name]) < 1e-8

def test_toy_iterations():
  net = init_toy_model()
  X, y = init_toy_data()
  net = init_toy_model()
  with np.errstate(divide='raise'):
    stats = net.train(X, y, X, y,
              learning_rate=1e-1, reg=1e-5,
              num_iters=100, verbose=False)


  final_training_loss = stats['loss_history'][-1]
  print 'Final training loss: ', final_training_loss
  assert final_training_loss < 1e1
