from typing import Union, List, Tuple
from PIL import Image

PYTORCH = "PyTorch"
TENSORFLOW = "Tensorflow"
NO_LABEL_PLACEHOLDER = "None"  # A place holder when only one label is provided
CIL_SETUP = "CIL"  # Class Incemental Learning Setup
IIRC_SETUP = "IIRC"  # Incremental Implicitly Refined Classification Setup
DatasetStructType = Union[List[Tuple[Image.Image, Tuple[str, ...]]], List[Tuple[str, Tuple[str, ...]]]]