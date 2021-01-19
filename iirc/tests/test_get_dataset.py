import pytest
import numpy as np
from PIL import Image
from iirc.datasets_loader import _get_dataset

# Test that the output dataset have the specified shape (image/image_path, (label1,)) or
# (image/image_path, (label1, label2))
np.random.seed(100)


@pytest.mark.parametrize("dataset_name", ["iirc_cifar100", "iirc_imagenet_full"])
def test_get_dataset(imagenet_root, dataset_name):
    if dataset_name == "iirc_cifar100":
        datasets, dataset_configuration = \
            _get_dataset(dataset_name=dataset_name, dataset_root="./data")
        assert "superclass_data_pct" in dataset_configuration.keys()
        assert "subclass_data_pct" in dataset_configuration.keys()
        assert "intask_valid_train_ratio" in dataset_configuration.keys()
        assert "posttask_valid_train_ratio" in dataset_configuration.keys()
        assert "using_image_path" in dataset_configuration.keys()
    elif dataset_name == "iirc_imagenet_full":
        datasets, dataset_configuration = \
            _get_dataset(dataset_name=dataset_name, dataset_root=imagenet_root)
        assert "intask_valid_train_ratio" in dataset_configuration.keys()
        assert "posttask_valid_train_ratio" in dataset_configuration.keys()
        assert "using_image_path" in dataset_configuration.keys()

    assert "train" in datasets.keys()
    assert "intask_valid" in datasets.keys()
    assert "posttask_valid" in datasets.keys()
    assert "test" in datasets.keys()
    for dataset in datasets.values():
        sample_size = min(len(dataset), 100)
        sample_idx = np.arange(len(dataset))
        np.random.shuffle(sample_idx)
        sample_idx = sample_idx[:sample_size]
        for i in sample_idx:
            sample_image, sample_labels = dataset[i]
            assert len(sample_labels) == 1 or len(sample_labels) == 2
            assert isinstance(sample_image, Image.Image) or isinstance(sample_image,
                                                                       str)  # string in case of image path
