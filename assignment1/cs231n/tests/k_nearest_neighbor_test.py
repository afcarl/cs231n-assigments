"""Tests"""
import numpy as np
import pytest

from cs231n.classifiers import KNearestNeighbor

@pytest.fixture
def classifier():
  """Our memoized classifier"""
  train_data = np.array([
    [1, 2, 2, 1],
    [4, 3, 4, 4],
    [3, 4, 4, 2]
    ])

  train_labels = np.array([1, 2, 2])

  knn = KNearestNeighbor()
  knn.train(train_data, train_labels)
  return knn

def test_compute_distances_two_loops(classifier):
  """Make sure it works"""
  classifier.compute_distances_two_loops(np.array([1, 1, 1, 1]))
