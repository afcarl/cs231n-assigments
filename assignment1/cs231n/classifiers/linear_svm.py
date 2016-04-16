from random import shuffle

import numpy as np


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]

  loss = 0.0
  for i in xrange(num_train):
    x = X[i]
    scores = x.dot(W)
    correct_class = y[i]
    correct_class_score = scores[correct_class]
    for j in xrange(num_classes):
      # Compute the loss

      if j == correct_class:
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        incorrect_class_that_did_not_meet_margin = j
        # So we increase the loss
        loss += margin
        dW[:, correct_class] -= x
        # And we update the gradient for this class
        dW[:, incorrect_class_that_did_not_meet_margin] += x

    # Update the gradient for the weights that feed the correct class


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


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in .                                                           #
  #############################################################################
  scores = X.dot(W) # shape = (N, C) TODO: do this in the right direction
  correct_classes = [np.arange(y.shape[0]),
                     y] # shape = (2, N)

  correct_class_scores = scores[correct_classes] # shape (N)
  correct_class_scores = correct_class_scores[:, np.newaxis] # shape (N, 1)
  # import pytest; pytest.set_trace()

  margins = scores - correct_class_scores + 1 # shape (N, C)

  # Zero out the loss for all the correct classes
  margins[correct_classes] = 0

  # Zero out the loss for everything below zero
  losses = np.maximum(margins, 0)

  loss = np.sum(losses) / num_train
  loss += 0.5 * reg * np.sum(np.square(W))

  # Put 1 in for each [training point, class] pair above the margin
  classes_beyond_margin = (margins > 0).astype(np.float64) # shape (N, C)

  # For the correct classes, replace the value with the count the count
  # of classes above the margin
  num_classes_beyond_margin = np.sum((classes_beyond_margin), axis=1) # shape N
  classes_beyond_margin[correct_classes] = -num_classes_beyond_margin

  # Dot-product each row with our X
  dW = X.T.dot(classes_beyond_margin)

  dW /= num_train
  dW += reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
