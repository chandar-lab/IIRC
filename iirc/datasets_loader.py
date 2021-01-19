import os
import numpy as np
import json
from typing import Any, Optional, Callable, List, Tuple, Dict
from PIL import Image

from iirc.definitions import PYTORCH, TENSORFLOW, CIL_SETUP, IIRC_SETUP, DatasetStructType
from iirc.utils.utils import unpickle, print_msg
from iirc.lifelong_dataset.base_dataset import BaseDataset

metadata_root = os.path.join(os.path.dirname(__file__), "./metadata")
datasets_names = ["incremental_cifar100", "iirc_cifar100", "incremental_imagenet_full", "incremental_imagenet_lite",
                  "iirc_imagenet_full", "iirc_imagenet_lite"]


def get_lifelong_datasets(dataset_name: str,
                          dataset_root: str = "./data",
                          setup: str = IIRC_SETUP,
                          framework: str = PYTORCH,
                          tasks_configuration_id: int = 0,
                          essential_transforms_fn: Optional[Callable[[Image.Image], Any]] = None,
                          augmentation_transforms_fn: Optional[Callable[[Image.Image], Any]] = None,
                          cache_images: bool = False, joint: bool = False) -> Tuple[
    Dict[str, BaseDataset], List[List[str]], Dict[str, int]]:
    """
    Get the incremental refinement learning , as well as the tasks (which contains the classes introduced at each task),
    and the index for each class corresponding to its order of appearance

    Args:
        dataset_name (str): The name of the dataset, ex: iirc_cifar100
        dataset_root (str): The directory where the dataset is/will be downloaded (default: "./data")
        setup (str): Class Incremental Learning setup (CIL) or Incremental Implicitly Refined Classification setup
            (IIRC) (default: IIRC_SETUP)
        framework (str): The framework to be used, whether PyTorch or Tensorflow. use Tensorflow for any numpy based
            dataloading  (default: PYTORCH)
        tasks_configuration_id (int): The configuration id, where each configuration corresponds to a specific tasks and
            classes order for each dataset. This id starts from 0 for each dataset. Ignore when joint is set to True
            (default: 0)
        essential_transforms_fn (Optional[Callable[[Image.Image], Any]]): A function that contains the essential
            transforms (for example, converting a pillow image to a tensor) that should be applied to each image. This
            function is applied only when the augmentation_transforms_fn is set to None (as in the case of a test set)
            or inside the disable_augmentations context (default: None)
        augmentation_transforms_fn: A function that contains the essential transforms (for example, converting a pillow
            image to a tensor) and augmentation transforms (for example, applying random cropping) that should be
            applied to each image. When this function is provided, essential_transforms_fn is not used except inside the
            disable_augmentations context (default: None)
        cache_images (bool): cache images that belong to the current task in the memory, only applicable when using the
            image path (default: False)
        joint (bool): provided all the classes in a single task for joint training (default: False)

    Returns:
        Tuple[Dict[str, BaseDataset], List[List[str]], Dict[str, int]]:

        lifelong_datasets (Dict[str, BaseDataset]): a dictionary with the keys corresponding to the four splits (train,
        intask_validation, posttask_validation, test), and the values containing the dataset object inheriting from
        BaseDataset for that split.

        tasks (List[List[str]]): a list of lists where each inner list contains the set of classes (class names) that
        will be introduced in that task (example: [[dog, cat, car], [tiger, truck, fish]]).

        class_names_to_idx (Dict[str, int]): a dictionary with the class name as key, and the class index as value
        (example: {"dog": 0, "cat": 1, ...}).
    """
    assert framework in [PYTORCH, TENSORFLOW], f'The framework is set to neither "{PYTORCH}" nor "{TENSORFLOW}"'
    assert setup in [IIRC_SETUP, CIL_SETUP], f'The setup is set to neither "{IIRC_SETUP}" nor "{CIL_SETUP}"'
    assert dataset_name in datasets_names, f'The dataset_name is not in {datasets_names}'
    print_msg(f"Creating {dataset_name}")

    datasets, dataset_configuration = \
        _get_dataset(dataset_name=dataset_name, dataset_root=dataset_root)
    tasks, class_names_to_idx = _get_tasks_configuration(dataset_name, tasks_configuration_id=tasks_configuration_id,
                                                         joint=joint)

    sprcla_data_pct = dataset_configuration["superclass_data_pct"]
    subcla_data_pct = dataset_configuration["subclass_data_pct"]
    using_image_path = dataset_configuration["using_image_path"]
    sprcla_sampling_size_cap = dataset_configuration["superclass_sampling_size_cap"]
    lifelong_datasets = {}

    if framework == PYTORCH:
        from iirc.lifelong_dataset.torch_dataset import Dataset
        LifeLongDataset = Dataset
    elif framework == TENSORFLOW:
        from iirc.lifelong_dataset.tensorflow_dataset import Dataset
        LifeLongDataset = Dataset
    else:
        raise NotImplementedError

    print_msg(f"Setup used: {setup}\nUsing {framework}")

    shared_arguments = dict(tasks=tasks, setup=setup, using_image_path=using_image_path,
                            cache_images=cache_images, essential_transforms_fn=essential_transforms_fn,
                            superclass_data_pct=sprcla_data_pct, subclass_data_pct=subcla_data_pct,
                            superclass_sampling_size_cap=sprcla_sampling_size_cap)
    lifelong_datasets["train"] = LifeLongDataset(dataset=datasets["train"], test_mode=False,
                                                 augmentation_transforms_fn=augmentation_transforms_fn,
                                                 **shared_arguments)
    lifelong_datasets["intask_valid"] = LifeLongDataset(dataset=datasets["intask_valid"], test_mode=False,
                                                        **shared_arguments)
    lifelong_datasets["posttask_valid"] = LifeLongDataset(dataset=datasets["posttask_valid"], test_mode=True,
                                                          **shared_arguments)
    lifelong_datasets["test"] = LifeLongDataset(dataset=datasets["test"], test_mode=True, **shared_arguments)

    print_msg("Dataset created")

    return lifelong_datasets, tasks, class_names_to_idx


def _get_dataset(dataset_name: str, dataset_root: str) -> Tuple[Dict[str, DatasetStructType], Dict]:
    """
    Loads the dataset using the DatasetTypeStruct structure and loads the dataset configuration (the
    superclass_data_pct, intask_valid_train_ratio, etc, for that specific dataset)

    Args:
        dataset_name (str): The name of the dataset, ex: iirc_cifar100
        dataset_root (str): The directory where the dataset is/will be downloaded

    Returns:
        Tuple[Dict[str, DatasetStructType], Dict]:
        datasets (Dict[str, DatasetStructType]): a dictionary with the keys corresponding to the four splits (train,
        intask_validation, posttask_validation, test), and the values being a list of the samples that belong to
        each split (with the images or images paths) in the DatasetTypeStruct structure
        dataset_configuration (Dict): a dictionary with the configuration corresponding to this dataset (the
        superclass_data_pct, intask_valid_train_ratio, etc)
    """
    with open(os.path.join(metadata_root, "dataset_configurations.json"), "r") as f:
        dataset_configuration = json.load(f)
    if dataset_name == "iirc_cifar100":
        dataset_configuration = dataset_configuration["iirc_cifar100"]

        datasets = \
            _load_iirc_cifar100(root=dataset_root,
                                intask_valid_train_ratio=dataset_configuration["intask_valid_train_ratio"],
                                posttask_valid_train_ratio=dataset_configuration["posttask_valid_train_ratio"])
    elif dataset_name == "incremental_cifar100":
        dataset_configuration = dataset_configuration["incremental_cifar100"]
        datasets = \
            _load_incremental_cifar100(root=dataset_root,
                                       intask_valid_train_ratio=dataset_configuration["intask_valid_train_ratio"],
                                       posttask_valid_train_ratio=dataset_configuration["posttask_valid_train_ratio"])
    elif "iirc_imagenet" in dataset_name:
        dataset_configuration = dataset_configuration["iirc_imagenet"]
        datasets = \
            _load_iirc_imagenet(root=dataset_root,
                                intask_valid_train_ratio=dataset_configuration["intask_valid_train_ratio"],
                                posttask_valid_train_ratio=dataset_configuration["posttask_valid_train_ratio"])
    elif "incremental_imagenet" in dataset_name:
        dataset_configuration = dataset_configuration["iirc_imagenet"]
        datasets = \
            _load_incremental_imagenet(root=dataset_root,
                                       intask_valid_train_ratio=dataset_configuration["intask_valid_train_ratio"],
                                       posttask_valid_train_ratio=dataset_configuration["posttask_valid_train_ratio"])
    else:
        raise ValueError(f"The dataset {dataset_name} is not implemented (or check the spelling)")
    return datasets, dataset_configuration


def _load_iirc_cifar100(root: str = ".", intask_valid_train_ratio: float = 0.1,
                        posttask_valid_train_ratio: float = 0.1) -> Dict[str, DatasetStructType]:
    """
    Load CIFAR100 dataset and convert it to IIRC-CIFAR100 format

    Args:
        root (string): The location of the dataset
        intask_valid_train_ratio (float): the percentage of the training set to be taken for the in-task validation set
            , a training-like validation set used for valdation during the task training (default: 0.1)
        posttask_valid_train_ratio (float): the percentage of the training set to be taken for the post-task validation
            set, a test-like validation set used for valdation after the task training (default: 0.1)

    Returns:
        Dict[str, DatasetStructType]: datasets, a dictionary with the keys corresponding to the four splits (train,
        intask_validation, posttask_validation, test), and the values being a list of the samples that belong to
        each split (with the images provided in Image.Image type) in the DatasetTypeStruct structure
    """
    raw_data_train = unpickle(os.path.join(root, "cifar-100-python", "train"))
    raw_data_test = unpickle(os.path.join(root, "cifar-100-python", "test"))
    raw_data_meta = unpickle(os.path.join(root, "cifar-100-python", "meta"))

    with open(os.path.join(metadata_root, "iirc_cifar100_hierarchy.json"), "r") as f:
        class_hierarchy = json.load(f)

    datasets = {"train": [], "intask_valid": [], "posttask_valid": [], "test": []}

    class_to_idx = {class_.decode('UTF-8'): i for class_, i in zip(raw_data_meta[b'fine_label_names'], range(100))}
    train_targets = np.array(raw_data_train[b'fine_labels'])
    test_targets = np.array(raw_data_test[b'fine_labels'])

    superclass_to_subclasses = class_hierarchy['super_classes']
    subclasses_to_superclass = {subclass: superclass for superclass, subclasses in superclass_to_subclasses.items()
                                for subclass in subclasses}
    for subclass in class_hierarchy['other_sub_classes']:
        subclasses_to_superclass[subclass] = None

    for subclass in subclasses_to_superclass.keys():
        samples_idx = {"train": [], "intask_valid": [], "posttask_valid": [], "test": []}
        superclass = subclasses_to_superclass[subclass]
        subclass_id = class_to_idx[subclass]
        samples_idx["test"] = np.argwhere(test_targets == subclass_id).squeeze(1)
        samples_idx["train"] = np.argwhere(train_targets == subclass_id).squeeze(1)

        original_train_len = len(samples_idx["train"])
        intask_valid_offset = int(intask_valid_train_ratio * original_train_len)
        posttask_valid_offset = int(posttask_valid_train_ratio * original_train_len) + intask_valid_offset
        samples_idx["intask_valid"] = samples_idx["train"][:intask_valid_offset]
        samples_idx["posttask_valid"] = samples_idx["train"][intask_valid_offset:posttask_valid_offset]
        samples_idx["train"] = samples_idx["train"][posttask_valid_offset:]

        assert "test" in datasets.keys()
        for dataset_type in datasets.keys():
            if dataset_type == "test":
                raw_data = raw_data_test
            else:
                raw_data = raw_data_train
            for idx in samples_idx[dataset_type]:
                image = raw_data[b'data'][idx].reshape((3, 32, 32)).transpose(1, 2, 0)
                image = Image.fromarray(image)
                if superclass is None:
                    datasets[dataset_type].append((image, (subclass,)))
                else:
                    datasets[dataset_type].append((image, (superclass, subclass)))
    return datasets


def _load_incremental_cifar100(root: str = ".", intask_valid_train_ratio: float = 0.1,
                               posttask_valid_train_ratio: float = 0.1) -> Dict[str, DatasetStructType]:
    """
    Load CIFAR100 dataset and convert it to incremental CIFAR100 format (for class incremental learning)

    Args:
        root (string): The location of the dataset
        intask_valid_train_ratio (float): the percentage of the training set to be taken for the in-task validation set
            , a training-like validation set used for valdation during the task training (default: 0.1)
        posttask_valid_train_ratio (float): the percentage of the training set to be taken for the post-task validation
            set, a test-like validation set used for valdation after the task training (default: 0.1)

    Returns:
        Dict[str, DatasetStructType]: datasets, a dictionary with the keys corresponding to the four splits (train,
        intask_validation, posttask_validation, test), and the values being a list of the samples that belong to
        each split (with the images provided in Image.Image type) in the DatasetTypeStruct structure
    """
    raw_data_train = unpickle(os.path.join(root, "cifar-100-python", "train"))
    raw_data_test = unpickle(os.path.join(root, "cifar-100-python", "test"))
    raw_data_meta = unpickle(os.path.join(root, "cifar-100-python", "meta"))

    with open(os.path.join(metadata_root, "cifar100_classes.json"), "r") as f:
        classes = json.load(f)

    datasets = {"train": [], "intask_valid": [], "posttask_valid": [], "test": []}

    class_to_idx = {class_.decode('UTF-8'): i for class_, i in zip(raw_data_meta[b'fine_label_names'], range(100))}
    train_targets = np.array(raw_data_train[b'fine_labels'])
    test_targets = np.array(raw_data_test[b'fine_labels'])

    classes = classes["classes"]

    for class_ in classes:
        samples_idx = {"train": [], "intask_valid": [], "posttask_valid": [], "test": []}
        subclass_id = class_to_idx[class_]
        samples_idx["test"] = np.argwhere(test_targets == subclass_id).squeeze(1)
        samples_idx["train"] = np.argwhere(train_targets == subclass_id).squeeze(1)

        original_train_len = len(samples_idx["train"])
        intask_valid_offset = int(intask_valid_train_ratio * original_train_len)
        posttask_valid_offset = int(posttask_valid_train_ratio * original_train_len) + intask_valid_offset
        samples_idx["intask_valid"] = samples_idx["train"][:intask_valid_offset]
        samples_idx["posttask_valid"] = samples_idx["train"][intask_valid_offset:posttask_valid_offset]
        samples_idx["train"] = samples_idx["train"][posttask_valid_offset:]

        assert "test" in datasets.keys()
        for dataset_type in datasets.keys():
            if dataset_type == "test":
                raw_data = raw_data_test
            else:
                raw_data = raw_data_train
            for idx in samples_idx[dataset_type]:
                image = raw_data[b'data'][idx].reshape((3, 32, 32)).transpose(1, 2, 0)
                image = Image.fromarray(image)
                datasets[dataset_type].append((image, (class_,)))
    return datasets


def _load_iirc_imagenet(root: str = ".", intask_valid_train_ratio: float = 0.04,
                        posttask_valid_train_ratio: float = 0.04) -> Dict[str, DatasetStructType]:
    """
    Load Imagenet dataset and convert it to IIRC-ImageNet format

    Args:
        root (string): The location of the dataset
        intask_valid_train_ratio (float): the percentage of the training set to be taken for the in-task validation set
            , a training-like validation set used for valdation during the task training (default: 0.1)
        posttask_valid_train_ratio (float): the percentage of the training set to be taken for the post-task validation
            set, a test-like validation set used for valdation after the task training (default: 0.1)

    Returns:
        Dict[str, DatasetStructType]: datasets, a dictionary with the keys corresponding to the four splits (train,
        intask_validation, posttask_validation, test), and the values being a list of the samples that belong to
        each split (with the images paths provided so that images can be loaded on the fly) in the DatasetTypeStruct
        structure
    """
    with open(os.path.join(metadata_root, "iirc_imagenet_hierarchy_wnids.json"), "r") as f:
        class_hierarchy = json.load(f)

    subclasses_wnids = []
    subclasses_wnids.extend(class_hierarchy["other_sub_classes"])
    for superclass in class_hierarchy["super_classes"]:
        subclasses_wnids.extend(class_hierarchy["super_classes"][superclass])
    assert len(subclasses_wnids) == 998

    datasets = {"train": [], "intask_valid": [], "posttask_valid": [], "test": []}

    class_to_idx = {subclass_: i for subclass_, i in zip(subclasses_wnids, range(998))}

    train_data_path = os.path.join(root, "train")
    test_data_path = os.path.join(root, "val")
    train_data = []
    train_targets = []
    test_data = []
    test_targets = []

    assert set(subclasses_wnids).issubset(set(os.listdir(train_data_path))), \
        f"classes {set(subclasses_wnids) - set(os.listdir(train_data_path))} are not in the training folder"
    for subfolder in subclasses_wnids:
        subclass_path = os.path.join(train_data_path, subfolder)
        class_idx = class_to_idx[subfolder]
        files = sorted(os.listdir(subclass_path))
        for file in files:
            assert file.split(".")[-1] == "JPEG", f"The samples files {file} are not of the correct format"
            train_data.append(os.path.join(subclass_path, file))
            train_targets.append(class_idx)

    assert set(subclasses_wnids).issubset(set(os.listdir(test_data_path))), \
        f"classes {set(subclasses_wnids) - set(os.listdir(test_data_path))} are not in the test folder"
    for subfolder in subclasses_wnids:
        subclass_path = os.path.join(test_data_path, subfolder)
        class_idx = class_to_idx[subfolder]
        files = sorted(os.listdir(subclass_path))
        for file in files:
            assert file.split(".")[-1] == "JPEG", f"The samples files {file} are not of the correct format"
            test_data.append(os.path.join(subclass_path, file))
            test_targets.append(class_idx)

    train_sort_indices = sorted(range(len(train_data)), key=lambda k: train_data[k])
    train_data = [file for _, file in sorted(zip(train_sort_indices, train_data), key=lambda pair: pair[0])]
    train_targets = [target for _, target in sorted(zip(train_sort_indices, train_targets), key=lambda pair: pair[0])]
    train_targets = np.array(train_targets)

    test_sort_indices = sorted(range(len(test_data)), key=lambda k: test_data[k])
    test_data = [file for _, file in sorted(zip(test_sort_indices, test_data), key=lambda pair: pair[0])]
    test_targets = [target for _, target in sorted(zip(test_sort_indices, test_targets), key=lambda pair: pair[0])]
    test_targets = np.array(test_targets)

    superclass_to_subclasses = class_hierarchy['super_classes']
    subclasses_to_superclass = {subclass: superclass for superclass, subclasses in superclass_to_subclasses.items()
                                for subclass in subclasses}
    for subclass in class_hierarchy['other_sub_classes']:
        assert subclass not in subclasses_to_superclass.keys(), f"{subclass} is repeated in the hierarchy"
        subclasses_to_superclass[subclass] = None

    for subclass in subclasses_to_superclass.keys():
        samples_idx = {"train": [], "intask_valid": [], "posttask_valid": [], "test": []}
        superclass = subclasses_to_superclass[subclass]
        subclass_id = class_to_idx[subclass]
        samples_idx["test"] = np.argwhere(test_targets == subclass_id).squeeze(1)
        samples_idx["train"] = np.argwhere(train_targets == subclass_id).squeeze(1)

        original_train_len = len(samples_idx["train"])
        intask_valid_offset = int(intask_valid_train_ratio * original_train_len)
        posttask_valid_offset = int(posttask_valid_train_ratio * original_train_len) + intask_valid_offset
        samples_idx["intask_valid"] = samples_idx["train"][:intask_valid_offset]
        samples_idx["posttask_valid"] = samples_idx["train"][intask_valid_offset:posttask_valid_offset]
        samples_idx["train"] = samples_idx["train"][posttask_valid_offset:]

        assert "test" in datasets.keys()
        for dataset_type in datasets.keys():
            if dataset_type == "test":
                raw_data = test_data
            else:
                raw_data = train_data
            for idx in samples_idx[dataset_type]:
                image_path = raw_data[idx]
                if superclass is None:
                    datasets[dataset_type].append((image_path, (subclass,)))
                else:
                    datasets[dataset_type].append((image_path, (superclass, subclass)))
    return datasets


def _load_incremental_imagenet(root: str = ".", intask_valid_train_ratio: float = 0.04,
                               posttask_valid_train_ratio: float = 0.04) -> Dict[str, DatasetStructType]:
    """
    Load Imagenet dataset and convert it to incremental Imagenet format (for class incremental learning)

    Args:
        root (string): The location of the dataset
        intask_valid_train_ratio (float): the percentage of the training set to be taken for the in-task validation set
            , a training-like validation set used for valdation during the task training (default: 0.1)
        posttask_valid_train_ratio (float): the percentage of the training set to be taken for the post-task validation
            set, a test-like validation set used for valdation after the task training (default: 0.1)

    Returns:
        Dict[str, DatasetStructType]: datasets, a dictionary with the keys corresponding to the four splits (train,
        intask_validation, posttask_validation, test), and the values being a list of the samples that belong to
        each split (with the images paths provided so that images can be loaded on the fly) in the DatasetTypeStruct
        structure
    """
    with open(os.path.join(metadata_root, "imagenet_classes.json"), "r") as f:
        classes_wnids = json.load(f)
    classes_wnids = classes_wnids["classes"]
    assert len(classes_wnids) == 1000

    datasets = {"train": [], "intask_valid": [], "posttask_valid": [], "test": []}

    class_to_idx = {class_: i for class_, i in zip(classes_wnids, range(1000))}

    train_data_path = os.path.join(root, "train")
    test_data_path = os.path.join(root, "val")
    train_data = []
    train_targets = []
    test_data = []
    test_targets = []

    assert set(classes_wnids).issubset(set(os.listdir(train_data_path))), \
        f"classes {set(classes_wnids) - set(os.listdir(train_data_path))} are not in the training folder"
    for subfolder in classes_wnids:
        subclass_path = os.path.join(train_data_path, subfolder)
        class_idx = class_to_idx[subfolder]
        files = sorted(os.listdir(subclass_path))
        for file in files:
            assert file.split(".")[-1] == "JPEG", f"The samples files {file} are not of the correct format"
            train_data.append(os.path.join(subclass_path, file))
            train_targets.append(class_idx)

    assert set(classes_wnids).issubset(set(os.listdir(test_data_path))), \
        f"classes {set(classes_wnids) - set(os.listdir(test_data_path))} are not in the test folder"
    for subfolder in classes_wnids:
        subclass_path = os.path.join(test_data_path, subfolder)
        class_idx = class_to_idx[subfolder]
        files = sorted(os.listdir(subclass_path))
        for file in files:
            assert file.split(".")[-1] == "JPEG", f"The samples files {file} are not of the correct format"
            test_data.append(os.path.join(subclass_path, file))
            test_targets.append(class_idx)

    train_sort_indices = sorted(range(len(train_data)), key=lambda k: train_data[k])
    train_data = [file for _, file in sorted(zip(train_sort_indices, train_data), key=lambda pair: pair[0])]
    train_targets = [target for _, target in sorted(zip(train_sort_indices, train_targets), key=lambda pair: pair[0])]
    train_targets = np.array(train_targets)

    test_sort_indices = sorted(range(len(test_data)), key=lambda k: test_data[k])
    test_data = [file for _, file in sorted(zip(test_sort_indices, test_data), key=lambda pair: pair[0])]
    test_targets = [target for _, target in sorted(zip(test_sort_indices, test_targets), key=lambda pair: pair[0])]
    test_targets = np.array(test_targets)

    for class_ in classes_wnids:
        samples_idx = {"train": [], "intask_valid": [], "posttask_valid": [], "test": []}
        subclass_id = class_to_idx[class_]
        samples_idx["test"] = np.argwhere(test_targets == subclass_id).squeeze(1)
        samples_idx["train"] = np.argwhere(train_targets == subclass_id).squeeze(1)

        original_train_len = len(samples_idx["train"])
        intask_valid_offset = int(intask_valid_train_ratio * original_train_len)
        posttask_valid_offset = int(posttask_valid_train_ratio * original_train_len) + intask_valid_offset
        samples_idx["intask_valid"] = samples_idx["train"][:intask_valid_offset]
        samples_idx["posttask_valid"] = samples_idx["train"][intask_valid_offset:posttask_valid_offset]
        samples_idx["train"] = samples_idx["train"][posttask_valid_offset:]

        assert "test" in datasets.keys()
        for dataset_type in datasets.keys():
            if dataset_type == "test":
                raw_data = test_data
            else:
                raw_data = train_data
            for idx in samples_idx[dataset_type]:
                image_path = raw_data[idx]
                datasets[dataset_type].append((image_path, (class_,)))
    return datasets


def _get_tasks_configuration(dataset_name: str, tasks_configuration_id: int = 0, joint: bool = False) -> Tuple[
    List[List[str]], Dict[str, int]]:
    """
    Loads the tasks and classes order

    Args:
        dataset_name (str): The name of the dataset, ex: iirc_cifar100
        tasks_configuration_id (int): The configuration id, where each configuration corresponds to a specific tasks and
            classes order for each dataset. Ignore when joint is set to True (default: 0)
        joint (bool): provided all the classes in a single task for joint training (default: False)

    Returns:
        Tuple[List[List[str]], Dict[str, int]]:
        tasks (List[List[str]]): a list of lists where each inner list contains the set of classes (class names) that
        will be introduced in that task (example: [[dog, cat, car], [tiger, truck, fish]])
        class_names_to_idx (Dict[str, int]): a dictionary with the class name as key, and the class index as value
        (example: {"dog": 0, "cat": 1, ...})
    """
    if dataset_name == "iirc_cifar100":
        tasks_file = os.path.join(metadata_root, "iirc_cifar100_task_configurations.json")
        assert 0 <= tasks_configuration_id <= 9
    elif dataset_name == "incremental_cifar100":
        tasks_file = os.path.join(metadata_root, "incremental_cifar100_task_configurations.json")
        assert 0 <= tasks_configuration_id <= 9
    elif dataset_name == "iirc_imagenet_full":
        tasks_file = os.path.join(metadata_root, "iirc_imagenet_full_task_configurations.json")
        assert 0 <= tasks_configuration_id <= 4
    elif dataset_name == "iirc_imagenet_lite":
        tasks_file = os.path.join(metadata_root, "iirc_imagenet_lite_task_configurations.json")
        assert 0 <= tasks_configuration_id <= 4
    elif dataset_name == "incremental_imagenet_full":
        tasks_file = os.path.join(metadata_root, "incremental_imagenet_full_task_configurations.json")
        assert 0 <= tasks_configuration_id <= 4
    elif dataset_name == "incremental_imagenet_lite":
        tasks_file = os.path.join(metadata_root, "incremental_imagenet_lite_task_configurations.json")
        assert 0 <= tasks_configuration_id <= 4
    else:
        raise ValueError(f"The dataset {dataset_name} is not implemented (or check the spelling)")

    assert os.path.isfile(tasks_file)
    with open(tasks_file, "r") as f:
        tasks_configurations = json.load(f)
    tasks_configuration = tasks_configurations[f"configuration_{tasks_configuration_id}"]
    tasks = [tasks_configuration[f"task_{j}"] for j in range(len(tasks_configuration))]

    ordered_class_names = []
    for task in tasks:
        for class_name in task:
            if class_name not in ordered_class_names:
                ordered_class_names.append(class_name)
    class_names_to_idx = \
        {class_name: idx for class_name, idx in zip(ordered_class_names, np.arange(len(ordered_class_names)))}
    if joint:
        tasks = [ordered_class_names]
    return tasks, class_names_to_idx
