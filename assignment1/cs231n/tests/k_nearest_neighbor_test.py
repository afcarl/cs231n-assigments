import numpy as np
import pytest

from cs231n.classifiers import KNearestNeighbor

@pytest.fixture
def classifier():
  train_data = np.array(
    [
      [1, 2, 2, 1],
      [4, 3, 4, 4],
      [3, 4, 4, 2],
    ])

  train_labels = np.array([1, 2, 2])

  knn = KNearestNeighbor()
  knn.train(train_data, train_labels)
  return knn

def test_compute_distances(classifier):

  test_data = np.array(
    [
      [1, 2, 2, 1], # Perfect match of the first training point
      [3, 3, 2, 3],
    ])

  for distance_function in [
      classifier.compute_distances_two_loops,
      classifier.compute_distances_one_loop,
      classifier.compute_distances_no_loops,
    ]:
    distances = distance_function(test_data)

    assert distances.shape == (2, 3)
    assert distances[0, 0] == 0 # Exact match
    assert distances[0, 1] == 23
    assert distances[1, 2] == 6
