
import numpy as np
import pytest
from numpy.testing import assert_allclose

from cs231n.classifiers import softmax_loss_naive, softmax_loss_vectorized
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
def W_zeros():
  """
  - W: A numpy array of shape (D, C) containing weights.
  """
  return np.zeros((4, 2))

@pytest.fixture
def W_ones():
  """
  - W: A numpy array of shape (D, C) containing weights.
  """
  return np.zeros((4, 2)) + 1


def test_softmax_loss_naive(W_zeros, X, y):
  loss, gradient = softmax_loss_naive(W_zeros, X, y, reg=0.0)
  assert loss == 0.69314718055994529

  desired_gradient = np.array([
    [2., -2.],
    [1.666667, -1.666667],
    [2., -2.],
    [1.666667, -1.666667]
  ])

  assert_allclose(desired_gradient, gradient, 0.00001)



# def test_softmax_loss_naive_with_weights(W_ones, X, y):
#   loss, gradient = softmax_loss_naive(W_ones, X, y, reg=1.0)
#   assert loss == 5.0
#
#   desired_gradient = np.array([
#     [3., -1.],
#     [2.666667, -0.666667],
#     [3., -1.],
#     [2.666667, -0.666667]])
#
#
#   assert_allclose(desired_gradient, gradient, 0.00001)
#
#
# def test_softmax_loss_vectorized(W_zeros, X, y):
#   loss, gradient = softmax_loss_vectorized(W_zeros, X, y, reg=1.0)
#   assert loss == 1
#
#   desired_gradient = np.array([
#     [2., -2.],
#     [1.666667, -1.666667],
#     [2., -2.],
#     [1.666667, -1.666667]
#   ])
#
#   assert_allclose(desired_gradient, gradient, 0.00001)
#
#
#
# def test_softmax_loss_vectorized_with_weights(W_ones, X, y):
#   loss, gradient = softmax_loss_vectorized(W_ones, X, y, reg=1.0)
#   assert loss == 5.0
#
#   desired_gradient = np.array([
#     [3., -1.],
#     [2.666667, -0.666667],
#     [3., -1.],
#     [2.666667, -0.666667]])
#
#
#   assert_allclose(desired_gradient, gradient, 0.00001)
