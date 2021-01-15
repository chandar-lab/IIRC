import pytest
from experiments.metrics import jaccard_sim, modified_jaccard_sim, strict_accuracy
import torch

target1 = torch.tensor([[1, 0, 0, 0], [1, 0, 1, 0]]).bool()

predictions11 = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]]).bool()
JS11 = 0
MJS11 = 0
acc11 = 0

predictions12 = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]]).bool()
JS12 = 0.75
MJS12 = 0.75
acc12 = 0.5

predictions13 = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]]).bool()
JS13 = 0.375
MJS13 = 0.15625
acc13 = 0

predictions14 = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]]).bool()
JS14 = 0.375
MJS14 = 0.16667
acc14 = 0

predictions15 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]]).bool()
JS15 = 0
MJS15 = 0
acc15 = 0


@pytest.mark.parametrize("target, predictions, expected_JS, expected_MJS, expected_acc",
                         [(target1, predictions11, JS11, MJS11, acc11),
                          (target1, predictions12, JS12, MJS12, acc12),
                          (target1, predictions13, JS13, MJS13, acc13),
                          (target1, predictions14, JS14, MJS14, acc14),
                          (target1, predictions15, JS15, MJS15, acc15)])
def test_metrics(target, predictions, expected_JS, expected_MJS, expected_acc):
    assert round(jaccard_sim(predictions, target), 5) == expected_JS
    assert round(modified_jaccard_sim(predictions, target), 5) == expected_MJS
    assert round(strict_accuracy(predictions, target), 5) == expected_acc
