import numpy as np
from random import shuffle
import pytest

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) # shape (D, C)

  #############################################################################
  # Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in xrange(num_train):
    x = X[i] # shape (D)
    scores = x.dot(W)
    correct_class = y[i]

    exp_scores = np.exp(scores)

    # Confidence in this class
    p = exp_scores / np.sum(exp_scores, axis=0) # shape (C)

    correct_class_score = p[correct_class]

    loss += - np.log(correct_class_score)

    # # Update the gradients
    #
    for j in xrange(num_classes):
      dW[:, j] += p[j] * x # dW total is p*x

    dW[:, correct_class] -= x # dW total is (-1 + p)*x,
    # More force in this direction if p was low

    #
    # dW[:, :] += x[:, np.newaxis]
    # dW[:, correct_class] -= x  # Undo it for the correct class

  # Right now the loss and dW are sums over all training examples, but we
  # want them to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss and the gradient.
  loss += 0.5 * reg * np.sum(np.square(W))

  # For each weight Wij, the partial derivative dWij of the loss function above is:
  #     0.5 * reg * 2.0 * Wij
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W) # shape (D, C)

  #############################################################################
  # Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################


  scores = X.dot(W) # shape = (N, C)
  correct_classes = [np.arange(y.shape[0]),
                     y] # shape = (2, N)


  exp_scores = np.exp(scores) # shape = (N, C)

  # Confidence in each class
  p = exp_scores / np.sum(exp_scores, axis=1)[:, np.newaxis] # shape (N, C)
  correct_class_scores = p[correct_classes] # shape (N)

  correct_class_scores = correct_class_scores[:, np.newaxis] # shape (N, 1)
  loss = - np.sum(np.log(correct_class_scores))

  # Update the gradients
  p[correct_classes] -= 1 # Subtract 1 from all the correct class ps
  dW = X.T.dot(p) # dW total is x dot p

  # Right now the loss and dW are sums over all training examples, but we
  # want them to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss and the gradient.
  loss += 0.5 * reg * np.sum(np.square(W))

  # For each weight Wij, the partial derivative dWij of the loss function above is:
  #     0.5 * reg * 2.0 * Wij
  dW += reg * W

  return loss, dW
