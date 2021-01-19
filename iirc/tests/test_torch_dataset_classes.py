import pytest
import random
from iirc.lifelong_dataset.torch_dataset import Dataset
from iirc.definitions import IIRC_SETUP, NO_LABEL_PLACEHOLDER, CIL_SETUP

# Create an example dataset
# 'a' is superclass of 'aa' and 'ab', etc
random.seed(100)
superclass_subclass = {'a': ['aa', 'ab'], 'b': ['ba', 'bb', 'bc'], 'c': ['ca', 'cb']}
subclasses_superclasses = {subclass: superclass for superclass, subclasses in superclass_subclass.items()
                           for subclass in subclasses}
other_subclasses = ['_d', '_e', '_f']
dataset = []
n_samples_per_subclass = 100
for subclass in list(subclasses_superclasses.keys()) + other_subclasses:
    if subclass in subclasses_superclasses.keys():
        superclass = subclasses_superclasses[subclass]
        data_subset = [(None, (superclass, subclass)) for _ in range(n_samples_per_subclass)]
    else:
        data_subset = [(None, (subclass,)) for _ in range(n_samples_per_subclass)]
    dataset.extend(data_subset)
random.shuffle(dataset)

superclass_data_pct = 0.4
subclass_data_pct = 0.8

tasks = [['a', 'b'], ['c', 'aa', '_d'], ['ab', 'ba', '_f']]


@pytest.mark.parametrize("expected_task_data_size", [[200, 260, 260]])
def test_Dataset_divide_data_across_tasks_IIRC_Train(expected_task_data_size):
    lifelong_dataset = Dataset(dataset, tasks, essential_transforms_fn=lambda x: x, setup=IIRC_SETUP, test_mode=False,
                               superclass_data_pct=superclass_data_pct, subclass_data_pct=subclass_data_pct)
    for task_id in range(len(tasks)):
        assert len(lifelong_dataset.task_id_to_data_idx[task_id]) == expected_task_data_size[task_id]


@pytest.mark.parametrize("expected_task_data_size", [[500, 400, 300]])
def test_Dataset_divide_data_across_tasks_IIRC_Test(expected_task_data_size):
    lifelong_dataset = Dataset(dataset, tasks, test_mode=True, setup=IIRC_SETUP, essential_transforms_fn=lambda x: x)
    for task_id in range(len(tasks)):
        assert len(lifelong_dataset.task_id_to_data_idx[task_id]) == expected_task_data_size[task_id]


@pytest.mark.parametrize("expected_task_data_size", [[500, 400, 300]])
def test_Dataset_divide_data_across_tasks_CIL(expected_task_data_size):
    lifelong_dataset = Dataset(dataset, tasks, setup=CIL_SETUP, essential_transforms_fn=lambda x: x)
    for task_id in range(len(tasks)):
        assert len(lifelong_dataset.task_id_to_data_idx[task_id]) == expected_task_data_size[task_id]


@pytest.mark.parametrize("expected_task_data_size", [[200, 260, 260]])
def test_choose_task(expected_task_data_size):
    lifelong_dataset = Dataset(dataset, tasks, essential_transforms_fn=lambda x: x, setup=IIRC_SETUP,
                               superclass_data_pct=superclass_data_pct, subclass_data_pct=subclass_data_pct)

    for task_id in range(len(tasks)):
        lifelong_dataset.choose_task(task_id)
        assert len(lifelong_dataset) == expected_task_data_size[task_id]
        assert set(lifelong_dataset.seen_classes) == set([cla for task in tasks[:task_id + 1] for cla in task])

        # test that the no labels from outside the current task are given (when not using complete information mode in
        # the case of IIRC)
        assert lifelong_dataset.cur_task == tasks[task_id]
        for i in range(len(lifelong_dataset)):
            image, label_1, label_2 = lifelong_dataset[i]
            # Check that only one label is given when not using the complete information mode (in IIRC)
            assert label_2 == NO_LABEL_PLACEHOLDER
            assert label_1 in tasks[task_id]


@pytest.mark.parametrize("expected_task_data_size, expected_data_up_to_size",
                         [([200, 260, 260], [200, 440, 660])])
def test_complete_information_mode(expected_task_data_size, expected_data_up_to_size):
    lifelong_dataset = Dataset(dataset, tasks, essential_transforms_fn=lambda x: x, setup=IIRC_SETUP,
                               superclass_data_pct=superclass_data_pct,
                               subclass_data_pct=subclass_data_pct)
    lifelong_dataset.enable_complete_information_mode()

    for task_id in range(len(tasks)):
        lifelong_dataset.choose_task(task_id)
        assert len(lifelong_dataset) == expected_task_data_size[task_id]
        assert set(lifelong_dataset.seen_classes) == set([cla for task in tasks[:task_id + 1] for cla in task])

        # test that all labels that have been observed so far are given
        assert lifelong_dataset.cur_task == tasks[task_id]
        for i in range(len(lifelong_dataset)):
            image, label_1, label_2 = lifelong_dataset[i]
            assert label_1 == NO_LABEL_PLACEHOLDER or label_1 in lifelong_dataset.seen_classes
            assert label_2 == NO_LABEL_PLACEHOLDER or label_2 in lifelong_dataset.seen_classes
            if label_1 in subclasses_superclasses.keys():
                assert label_2 == subclasses_superclasses[label_1]
            elif label_2 in subclasses_superclasses.keys():
                assert label_1 == subclasses_superclasses[label_2]

        lifelong_dataset.load_tasks_up_to(task_id)
        assert len(lifelong_dataset) == expected_data_up_to_size[task_id]
        assert set(lifelong_dataset.cur_task) == set([cla for task in tasks[:task_id + 1] for cla in task])
        for i in range(len(lifelong_dataset)):
            image, label_1, label_2 = lifelong_dataset[i]
            assert label_1 == NO_LABEL_PLACEHOLDER or label_1 in lifelong_dataset.seen_classes
            assert label_2 == NO_LABEL_PLACEHOLDER or label_2 in lifelong_dataset.seen_classes
            if label_1 in subclasses_superclasses.keys():
                assert label_2 == subclasses_superclasses[label_1]
            elif label_2 in subclasses_superclasses.keys():
                assert label_1 == subclasses_superclasses[label_2]


@pytest.mark.parametrize("class1, class2, class1_size0, class1_size01, class2_size1, class2_size01",
                         [("a", "aa", 80, 140, 80, 100)])
def test_get_shuffled_image_indices(class1, class2, class1_size0, class1_size01, class2_size1, class2_size01):
    lifelong_dataset = Dataset(dataset, tasks, essential_transforms_fn=lambda x: x, setup=IIRC_SETUP,
                               superclass_data_pct=superclass_data_pct, subclass_data_pct=subclass_data_pct)
    assert class1 in tasks[0]
    assert class2 in tasks[1]

    lifelong_dataset.choose_task(0)
    class1_indices = lifelong_dataset.get_image_indices_by_cla(class1)
    assert len(class1_indices) == class1_size0
    for idx in class1_indices:
        image, label_1, label_2 = lifelong_dataset[idx]
        assert label_1 == class1
        assert label_2 == NO_LABEL_PLACEHOLDER

    lifelong_dataset.choose_task(1)
    class2_indices = lifelong_dataset.get_image_indices_by_cla(class2)
    assert len(class2_indices) == class2_size1
    for idx in class2_indices:
        image, label_1, label_2 = lifelong_dataset[idx]
        assert label_1 == class2
        assert label_2 == NO_LABEL_PLACEHOLDER

    lifelong_dataset.enable_complete_information_mode()
    lifelong_dataset.load_tasks_up_to(1)
    class1_indices = lifelong_dataset.get_image_indices_by_cla(class1)
    class2_indices = lifelong_dataset.get_image_indices_by_cla(class2)
    assert len(class1_indices) == class1_size01
    assert len(class2_indices) == class2_size01
    for idx in class1_indices:
        image, label_1, label_2 = lifelong_dataset[idx]
        if label_1 == class1:
            assert label_2 == class2 or label_2 == NO_LABEL_PLACEHOLDER
        elif label_2 == class1:
            assert label_1 == class2 or label_1 == NO_LABEL_PLACEHOLDER
        else:
            raise ValueError(f"{class1} is not there")
    for idx in class2_indices:
        image, label_1, label_2 = lifelong_dataset[idx]
        if label_1 == class2:
            assert label_2 == class1
        elif label_2 == class2:
            assert label_1 == class1
        else:
            raise ValueError(f"{class2} is not there")
