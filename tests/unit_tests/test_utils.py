import unittest
from unittest import TestCase

from mcda.utils import *


class TestUtils(unittest.TestCase):

    @staticmethod
    def get_input_matrix_1() -> pd.DataFrame:
        data = {'ind1': [1, 2, 3], 'std1': [0.1, 0.2, 0.3], 'ind2': [5, 6, 7], 'std2': [0.1, 0.1, 0.1],
                'ind3': [9, 10, 11], 'std3': [0.1, 0.1, 12]}
        df = pd.DataFrame(data=data)

        return df

    @staticmethod
    def get_input_matrix_2() -> pd.DataFrame:
        data = {'ind1': [1, 2, 3], 'std1': [0.1, 0.2, 0.3], 'ind2': [5, 6, 7], 'std2': [0.1, 0.1, 0.1],
                'ind3': [9, 10, 11], 'std3': [0.1, 0.1, 11]}
        df = pd.DataFrame(data=data)

        return df

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


    def test_pop_indexed_elements(self):
        # Given
        indexes = [0,2,4]
        in_list = [1,2,3,4,5,6]

        # When
        out_list = pop_indexed_elements(indexes, in_list)
        expected_list = [2,4,6]

        # Then
        isinstance(out_list, list)
        TestCase.assertListEqual(self, out_list, expected_list)

    def test_check_averages_larger_std(self):
        # Given
        input_matrix_1 = TestUtils.get_input_matrix_1()
        input_matrix_2 = TestUtils.get_input_matrix_2()

        # When
        is_average_larger_than_std_1 = check_averages_larger_std(input_matrix_1)
        is_average_larger_than_std_2 = check_averages_larger_std(input_matrix_2)

        # Then
        isinstance(is_average_larger_than_std_1, bool)
        self.assertFalse(is_average_larger_than_std_1)
        isinstance(is_average_larger_than_std_2, bool)
        assert(is_average_larger_than_std_2)


