
import numpy as np
import pytest
from numpy.testing import assert_allclose

from cs231n.classifiers import svm_loss_naive, svm_loss_vectorized
"""
 N = 3
 C = 2
 D = 4
"""

@pytest.fixture
def X():
  """
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  """
  return np.array(
    [
      [1, 2, 2, 1],
      [4, 3, 4, 4],
      [3, 4, 4, 2],
    ])

@pytest.fixture
def y():
  """
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  """
  return np.array(
    [
      0,
      1,
      1,
    ])

@pytest.fixture
def W():
  """
  - W: A numpy array of shape (D, C) containing weights.
  """
  return np.zeros((4, 2))

@pytest.fixture
def desired_gradient():
  return np.array([
    [ 2.      , -2.      ],
    [ 1.666667, -1.666667],
    [ 2.      , -2.      ],
    [ 1.666667, -1.666667]
  ])

def test_svm_loss_naive(W, X, y, desired_gradient):
  loss, gradient = svm_loss_naive(W, X, y, reg=1.0)
  assert loss == 1

  assert_allclose(desired_gradient, gradient, 0.00001)


def test_svm_loss_vectorized(W, X, y, desired_gradient):
  loss, gradient = svm_loss_vectorized(W, X, y, reg=1.0)
  assert loss == 1

  assert_allclose(desired_gradient, gradient, 0.00001)
