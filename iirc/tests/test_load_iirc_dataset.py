import pytest
from iirc.datasets_loader import _load_iirc_cifar100, _load_iirc_imagenet
import json
import os

# test data integrity
@pytest.mark.parametrize("intask_valid_train_ratio", [0, 0.15])
@pytest.mark.parametrize("posttask_valid_train_ratio", [0, 0.15])
@pytest.mark.parametrize("dataset_root", ["./data"])
@pytest.mark.parametrize("metadata_root", ["./iirc/metadata"])
def test_load_iirc_cifar100(dataset_root, metadata_root, intask_valid_train_ratio, posttask_valid_train_ratio):
    datasets = _load_iirc_cifar100(root=dataset_root, intask_valid_train_ratio=intask_valid_train_ratio,
                                   posttask_valid_train_ratio=posttask_valid_train_ratio)
    with open(os.path.join(metadata_root, "iirc_cifar100_hierarchy.json"), "r") as f:
        class_hierarchy = json.load(f)
    superclasses = list(class_hierarchy["super_classes"].keys())
    subclasses = [subclass for subclasses_list in list(class_hierarchy["super_classes"].values())
                  for subclass in subclasses_list]
    subclasses.extend(class_hierarchy["other_sub_classes"])

    n_subclasses = 100
    assert len(subclasses) == n_subclasses

    num_intask_valid_samples = int(intask_valid_train_ratio * 500) * n_subclasses
    num_posttask_valid_samples = int(posttask_valid_train_ratio * 500) * n_subclasses
    assert len(datasets["train"]) == (50000 - num_intask_valid_samples - num_posttask_valid_samples)
    assert len(datasets["test"]) == 10000
    assert len(datasets["intask_valid"]) == num_intask_valid_samples
    assert len(datasets["posttask_valid"]) == num_posttask_valid_samples

    for superclass in superclasses:
        num_intask_valid_samples = int(intask_valid_train_ratio * 500) * len(
            class_hierarchy["super_classes"][superclass])
        num_posttask_valid_samples = int(posttask_valid_train_ratio * 500) * len(
            class_hierarchy["super_classes"][superclass])
        num_train_samples = 500 * len(class_hierarchy["super_classes"][superclass]) - num_intask_valid_samples - \
                            num_posttask_valid_samples
        num_test_samples = 100 * len(class_hierarchy["super_classes"][superclass])
        assert len([0 for sample in datasets["train"] if superclass in sample[1]]) == num_train_samples
        assert len([0 for sample in datasets["test"] if superclass in sample[1]]) == num_test_samples
        assert len([0 for sample in datasets["intask_valid"] if superclass in sample[1]]) == \
               num_intask_valid_samples
        assert len([0 for sample in datasets["posttask_valid"] if superclass in sample[1]]) == \
               num_posttask_valid_samples

    for subclass in subclasses:
        num_intask_valid_samples = int(intask_valid_train_ratio * 500)
        num_posttask_valid_samples = int(posttask_valid_train_ratio * 500)
        num_train_samples = 500 - num_intask_valid_samples - num_posttask_valid_samples
        num_test_samples = 100
        assert len([0 for sample in datasets['train'] if subclass in sample[1]]) == num_train_samples
        assert len([0 for sample in datasets["test"] if subclass in sample[1]]) == num_test_samples
        assert len([0 for sample in datasets["intask_valid"] if subclass in sample[1]]) == \
               num_intask_valid_samples
        assert len([0 for sample in datasets["posttask_valid"] if subclass in sample[1]]) == \
               num_posttask_valid_samples


@pytest.mark.parametrize("intask_valid_train_ratio", [0, 0.1])
@pytest.mark.parametrize("posttask_valid_train_ratio", [0, 0.1])
@pytest.mark.parametrize("metadata_root", ["./iirc/metadata"])
def test_load_iirc_imagenet(imagenet_root, metadata_root, intask_valid_train_ratio, posttask_valid_train_ratio):
    datasets = _load_iirc_imagenet(root=imagenet_root, intask_valid_train_ratio=intask_valid_train_ratio,
                                   posttask_valid_train_ratio=posttask_valid_train_ratio)
    with open(os.path.join(metadata_root, "iirc_imagenet_hierarchy_wnids.json"), "r") as f:
        class_hierarchy = json.load(f)
    superclasses = list(class_hierarchy["super_classes"].keys())
    subclasses = [subclass for subclasses_list in list(class_hierarchy["super_classes"].values())
                  for subclass in subclasses_list]
    subclasses.extend(class_hierarchy["other_sub_classes"])

    n_subclasses = 998
    assert len(subclasses) == n_subclasses

    num_intask_valid_samples_max = int(intask_valid_train_ratio * 1300) * n_subclasses
    num_intask_valid_samples_min = int(intask_valid_train_ratio * 1250) * n_subclasses
    num_posttask_valid_samples_max = int(posttask_valid_train_ratio * 1300) * n_subclasses
    num_posttask_valid_samples_min = int(posttask_valid_train_ratio * 1250) * n_subclasses
    assert (1250*n_subclasses - num_intask_valid_samples_min - num_posttask_valid_samples_min) \
           <= len(datasets["train"]) <= \
           (1300*n_subclasses - num_intask_valid_samples_max - num_posttask_valid_samples_max)
    assert len(datasets["test"]) == n_subclasses * 50
    assert num_intask_valid_samples_min <= len(datasets["intask_valid"]) <= num_intask_valid_samples_max
    assert num_posttask_valid_samples_min <= len(datasets["posttask_valid"]) <= num_posttask_valid_samples_max

    for superclass in superclasses:
        num_intask_valid_samples_min = int(intask_valid_train_ratio * 700) * len(
            class_hierarchy["super_classes"][superclass])
        num_posttask_valid_samples_min = int(posttask_valid_train_ratio * 700) * len(
            class_hierarchy["super_classes"][superclass])
        num_train_samples_min = 700 * len(class_hierarchy["super_classes"][superclass]) - \
                                num_intask_valid_samples_min - num_posttask_valid_samples_min
        num_test_samples_min = 50 * len(class_hierarchy["super_classes"][superclass])
        assert len([0 for sample in datasets["train"] if superclass in sample[1]]) >= num_train_samples_min
        assert len([0 for sample in datasets["test"] if superclass in sample[1]]) >= num_test_samples_min
        assert len([0 for sample in datasets["intask_valid"] if superclass in sample[1]]) >= \
               num_intask_valid_samples_min
        assert len([0 for sample in datasets["posttask_valid"] if superclass in sample[1]]) >= \
               num_posttask_valid_samples_min

    for subclass in subclasses:
        num_intask_valid_samples_min = int(intask_valid_train_ratio * 700)
        num_posttask_valid_samples_min = int(posttask_valid_train_ratio * 700)
        num_train_samples_min = 700 - num_intask_valid_samples_min - num_posttask_valid_samples_min
        num_test_samples_min = 50
        assert len([0 for sample in datasets['train'] if subclass in sample[1]]) >= num_train_samples_min
        assert len([0 for sample in datasets["test"] if subclass in sample[1]]) >= num_test_samples_min
        assert len([0 for sample in datasets["intask_valid"] if subclass in sample[1]]) >= \
               num_intask_valid_samples_min
        assert len([0 for sample in datasets["posttask_valid"] if subclass in sample[1]]) >= \
               num_posttask_valid_samples_min
