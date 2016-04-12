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

@pytest.fixture
def dists():
  return np.array([
    [0., 23., 13.],
    [9., 6., 6.]])

@pytest.fixture
def test_data():
  return np.array(
    [
      [1, 2, 2, 1], # Perfect match of the first training point
      [3, 3, 2, 3],
    ])

def test_compute_distances(classifier, dists, test_data):

  for distance_function in [
      classifier.compute_distances_two_loops,
      classifier.compute_distances_one_loop,
      classifier.compute_distances_no_loops,
    ]:
    computed_dists = distance_function(test_data)

    assert computed_dists.shape == (2, 3)
    assert computed_dists[0, 0] == 0 # Exact match

    assert np.array_equal(computed_dists, dists)


def test_predict_labels_with_k_as_one(classifier, dists):
  predicted_labels = classifier.predict_labels(dists, 1)
  assert predicted_labels.shape == (2,)
  assert predicted_labels[0] == 1
  assert predicted_labels[1] == 2

def test_predict_labels_with_k_as_two(classifier, dists):
  predicted_labels = classifier.predict_labels(dists, 2)
  assert predicted_labels.shape == (2,)
  assert predicted_labels[0] == 1
  assert predicted_labels[1] == 2


def test_predict_labels_with_k_as_three(classifier, dists):
  predicted_labels = classifier.predict_labels(dists, 3)
  assert predicted_labels.shape == (2,)
  assert predicted_labels[0] == 2
  assert predicted_labels[1] == 2
