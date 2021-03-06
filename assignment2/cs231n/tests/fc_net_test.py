import numpy as np
import pytest
from numpy.testing import assert_allclose
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

def assert_close(x, y, delta=1e-7):
  assert np.sum(np.abs(x - y)) < delta

def test_fc_net():
  N, D, H, C = 3, 5, 50, 7
  X = np.random.randn(N, D)
  y = np.random.randint(C, size=N)

  std = 1e-2
  model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

  print 'Testing initialization ... '
  W1_std = abs(model.params['W1'].std() - std)
  b1 = model.params['b1']
  W2_std = abs(model.params['W2'].std() - std)
  b2 = model.params['b2']
  assert W1_std < std / 10, 'First layer weights do not seem right'
  assert np.all(b1 == 0), 'First layer biases do not seem right'
  assert W2_std < std / 10, 'Second layer weights do not seem right'
  assert np.all(b2 == 0), 'Second layer biases do not seem right'

  print 'Testing test-time forward pass ... '
  model.params['W1'] = np.linspace(-0.7, 0.3, num=D*H).reshape(D, H)
  model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
  model.params['W2'] = np.linspace(-0.3, 0.4, num=H*C).reshape(H, C)
  model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
  X = np.linspace(-5.5, 4.5, num=N*D).reshape(D, N).T
  scores = model.loss(X)
  correct_scores = np.asarray(
    [[11.53165108,  12.2917344,   13.05181771,  13.81190102,  14.57198434, 15.33206765,  16.09215096],
     [12.05769098,  12.74614105,  13.43459113,  14.1230412,   14.81149128, 15.49994135,  16.18839143],
     [12.58373087,  13.20054771,  13.81736455,  14.43418138,  15.05099822, 15.66781506,  16.2846319 ]])
  scores_diff = np.abs(scores - correct_scores).sum()
  assert scores_diff < 1e-6, 'Problem with test-time forward pass'

  print 'Testing training loss (no regularization)'
  y = np.asarray([0, 5, 1])
  loss, grads = model.loss(X, y)
  correct_loss = 3.4702243556
  assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'

  model.reg = 1.0
  loss, grads = model.loss(X, y)
  correct_loss = 26.5948426952
  assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'

  for reg in [0.0, 0.7]:
    print 'Running numeric gradient check with reg = ', reg
    model.reg = reg
    loss, grads = model.loss(X, y)
    # print grads
    for name in sorted(grads):
      f = lambda _: model.loss(X, y)[0]
      grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)

      print name
      assert_close(grad_num, grads[name])


def test_generalized_FullyConnectedNet():
  N, D, H1, H2, C = 2, 15, 20, 30, 10
  X = np.random.randn(N, D)
  y = np.random.randint(C, size=(N,))

  for reg in [0, 3.14]:
    print 'Running check with reg = ', reg
    model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                              reg=reg, weight_scale=5e-2, dtype=np.float64)

    loss, grads = model.loss(X, y)
    print 'Initial loss: ', loss
    assert loss > 0

    for name in sorted(grads):
      f = lambda _: model.loss(X, y)[0]
      grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
      assert_close(grad_num, grads[name])
