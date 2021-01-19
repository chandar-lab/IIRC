import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Optional, Callable, List, Tuple, Union

from iirc.lifelong_dataset.base_dataset import BaseDataset
from iirc.definitions import IIRC_SETUP, DatasetStructType


class Dataset(BaseDataset):
    """
    An Lifelong Learning dataset that returns images as numpy arrays or tensorflow tensors

    Args:
        dataset (DatasetStructType): a list of tuples which contains the data in the form of (image, (label,)) or
            (image, (label1,label2)). The image path (str) can be provided instead if the images would be loaded on the
            fly (see the argument using_image_path). label is a string representing the class name
        tasks (List[List[str]]): a list of lists where each inner list contains the set of classes (class names) that
            will be introduced in that task (example: [[dog, cat, car], [tiger, truck, fish]])
        setup (str): Class Incremental Learning setup (CIL) or Incremental Implicitly Refined Classification setup
            (IIRC) (default: IIRC_SETUP)
        using_image_path (bool): whether the pillow image is provided in the dataset argument, or the image path that
            would be used later to load the image. set True if using the image path (default: False)
        cache_images (bool): cache images that belong to the current task in the memory, only applicable when using
            the image path (default: False)
        essential_transforms_fn (Optional[Callable[[Image.Image], Union[tf.Tensor, np.ndarray]]]): A function that
            contains the essential transforms (for example, converting a pillow image to a tensor) that should be
            applied to each image. This function is applied only when the augmentation_transforms_fn is set to None
            (as in the case of a test set) or inside the disable_augmentations context (default: None)
        augmentation_transforms_fn: (Optional[Callable[[Image.Image], Union[tf.Tensor, np.ndarray]]]): A function that
            contains the essential transforms (for example, converting a pillow image to a tensor) and augmentation
            transforms (for example, applying random cropping) that should be applied to each image.
            When this function is provided, essential_transforms_fn is not used except inside the disable_augmentations
            context (default: None)
        test_mode (bool): Whether this dataset is considered a training split or a test split. This info is only helpful
            when using the IIRC setup (default: False)
        complete_information_mode (bool): Whether the dataset is in complete information mode or incomplete information
            mode.
            This is only valid when using the IIRC setup.
            In the incomplete information mode, if a sample has two labels corresponding to a previous task and a
            current task (example: dog and Bulldog), only the label present in the current task is provided (Bulldog).
            In the complete information mode, both labels will be provided. In all cases, no label from a future task
            would be provided.
            When no value is set for complete_information_mode, this value is defaulted to the test_mode value (complete
            information during test mode only) (default: None)
        superclass_data_pct (float) : The percentage of samples sampled for each superclass from its consistuent
            subclasses.
            This is valid only when using the IIRC setup and when test_mode is set to False.
            For example, If the superclass "dog" has the subclasses "Bulldog" and "Whippet", and superclass_data_pct is
            set to 0.4, then 40% of each of the "Bulldog" samples and "Whippet" samples will be provided when training
            on the task that has the class "dog"  (default: 0.6)
        subclass_data_pct (float): The percentage of samples sampled for each subclass if it has a superclass.
            This is valid only when using the IIRC setup and when test_mode is set to False.
            For example, If the superclass "dog" has one of the subclasses as "Bulldog", and superclass_data_pct is
            set to 0.4 while subclass_data_pct is set to 0.8, then 40% of the "Bulldog" samples will be provided when
            training on the task that contains "dog", and 80% of the "Bulldog" samples will be provided when training on
            the task that contains "Bulldog". superclass_data_pct and subclass_data_pct don't need to sum to 1 as the
            samples can be repeated across tasks (in the previous example, 20% of the samples were repeated across the
            two tasks) (default: 0.6)
        superclass_sampling_size_cap (int): The number of subclasses a superclass should contain after which the number
            of samples doesn't increase anymore.
            This is valid only when using the IIRC setup and when test_mode is set to False.
            For example, If a superclass has 8 subclasses, with the superclass_data_pct set to 0.4, and
            superclass_sampling_size_cap set to 5, then superclass_data_pct for that specific superclass will be
            adjusted to 0.25 (5 / 8 * 0.4) (default: 100)
    """

    def __init__(self,
                 dataset: DatasetStructType,
                 tasks: List[List[str]],
                 setup: str = IIRC_SETUP,
                 using_image_path: bool = False,
                 cache_images: bool = False,
                 essential_transforms_fn: Optional[Callable[[Image.Image], Union[tf.Tensor, np.ndarray]]] = None,
                 augmentation_transforms_fn: Optional[Callable[[Image.Image], Union[tf.Tensor, np.ndarray]]] = None,
                 test_mode: bool = False,
                 complete_information_mode: Optional[bool] = None,
                 superclass_data_pct: float = 0.6,
                 subclass_data_pct: float = 0.6,
                 superclass_sampling_size_cap: int = 100):
        if essential_transforms_fn is None:
            essential_transforms_fn = np.asarray
        if augmentation_transforms_fn is None:
            augmentation_transforms_fn = essential_transforms_fn
        super(Dataset, self).__init__(dataset=dataset, tasks=tasks, setup=setup,
                                      using_image_path=using_image_path, cache_images=cache_images,
                                      essential_transforms_fn=essential_transforms_fn,
                                      augmentation_transforms_fn=augmentation_transforms_fn,
                                      test_mode=test_mode,
                                      complete_information_mode=complete_information_mode,
                                      superclass_data_pct=superclass_data_pct,
                                      subclass_data_pct=subclass_data_pct,
                                      superclass_sampling_size_cap=superclass_sampling_size_cap)

    def __getitem__(self, index) -> Tuple[Union[tf.Tensor, np.ndarray], str, str]:
        image, label_1, label_2 = self.get_item(index)
        if self._apply_augmentations:
            image = self.augmentation_transforms_fn(image)
        else:
            image = self.essential_transforms_fn(image)
        return image, label_1, label_2
