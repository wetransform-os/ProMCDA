import unittest
import pytest

from mcda.utils import *


class TestUtils(unittest.TestCase):

    def test_randomly_sample_ix_weight(self):
        # Given
        num_weights = 4
        index = 1
        num_runs = 10

        # When
        out = randomly_sample_ix_weight(num_weights,index,num_runs)

        # Then
        isinstance(out, list)
        assert len(out) == num_runs
        assert [len(out[i]) == num_weights for i in range(num_weights)]
        for j in range(num_runs):
            for i in range(num_weights):
                if i != index:
                    assert out[j][i] == 1
                else:
                    assert out[j][i] != 1


    def test_randomly_sample_all_weights(self):
        # Given
        num_weights = 4
        num_runs = 10

        # When
        out = randomly_sample_all_weights(num_weights,num_runs)

        # Then
        isinstance(out, list)
        assert len(out) == num_runs
        assert [len(out[i]) == num_weights for i in range(num_weights)]
        for j in range(num_runs):
            assert (elem != out[0] for elem in out) # check all items are different
            for i in range(num_weights):
                    assert out[j][i] <= 1
                    assert out[j][i] >= 0


    def test_check_norm_sum_weights(self):
        # Given
        weights = [1,2,3,4,5]

        # When
        out = check_norm_sum_weights(weights)

        # Then
        assert len(out) == len(weights)
        assert sum(out) == 1.0