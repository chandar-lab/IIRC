import pytest
from experiments.utils import extend_list


@pytest.mark.parametrize("input_, output_length, expected_output",
                         [(5, 0, []),
                          ([10, 3], 0, []),
                          (5, 1, [5]),
                          (5, 2, [5, 5]),
                          ([10, 3, 2], 5, [10, 3, 2, 2, 2]),
                          ([10, 3, 2], 2, [10, 3])])
def test_extend_list(input_, output_length, expected_output):
    extended_list = extend_list(input_, output_length)
    assert extended_list == expected_output
