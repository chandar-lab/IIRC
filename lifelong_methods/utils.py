from copy import deepcopy
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, LambdaLR
import torch.nn as nn
import torch
import numpy as np
from torch.utils import data as data
from typing import Iterator, Iterable, Optional, Tuple, Union, List, Dict, TYPE_CHECKING

from iirc.definitions import NO_LABEL_PLACEHOLDER
from iirc.lifelong_dataset.torch_dataset import Dataset

if TYPE_CHECKING:
    from lifelong_methods.buffer.buffer import BufferBase
    from lifelong_methods.methods.base_method import BaseMethod


class SubsetSampler(data.Sampler):
    """
    Samples elements in order from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


def get_optimizer(model_parameters: Iterator[nn.parameter.Parameter], optimizer_type: str = "momentum",
                  lr: float = 0.01, lr_gamma: float = 1., lr_schedule: Optional[List[int]] = None,
                  reduce_lr_on_plateau: bool = False, weight_decay: float = 0.0001) -> Tuple[
    optim.Optimizer, Union[MultiStepLR, ReduceLROnPlateau, LambdaLR]]:
    """
    A method that returns the optimizer and scheduler to be used

    Args:
        model_parameters (Iterator[nn.parameter.Parameter]): the list of model parameters
        optimizer_type (string): the optimizer type to be used (currently only "momentum" and "adam" are supported)
        lr (float): The initial learning rate for each task
        lr_gamma (float): The multiplicative factor for learning rate decay at the epochs specified
        lr_schedule (Optional[List[int]]): the epochs per task at which to multiply the current learning rate by lr_gamma
            (resets after each task)
        reduce_lr_on_plateau (bool): reduce the lr on plateau based on the validation performance metric. If set to True,
            the lr_schedule is ignored
        weight_decay (float): The weight decay multiplier

    Returns:
        Tuple[optim.Optimizer, Union[MultiStepLR, ReduceLROnPlateau, LambdaLR]]:
        optimizer (optim.Optimizer):
        scheduler (Union[MultiStepLR, ReduceLROnPlateau, LambdaLR]):
     """
    if optimizer_type == "momentum":
        if lr is None:
            lr = 0.01
        optimizer = optim.SGD(model_parameters, lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        if lr is None:
            lr = 0.001
        optimizer = optim.Adam(model_parameters, lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"The optimizer type {optimizer_type} is not supported")

    if reduce_lr_on_plateau:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=lr_gamma, patience=10, verbose=True, threshold=1e-3,
                                      threshold_mode="abs")
    elif lr_schedule is not None:
        scheduler = MultiStepLR(optimizer, milestones=lr_schedule, gamma=lr_gamma)
    else:
        # a do nothing scheduler
        lr_lambda = lambda epoch: 1.0
        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    return optimizer, scheduler


def labels_index_to_one_hot(labels: torch.Tensor, length: int) -> torch.Tensor:
    labels = nn.functional.one_hot(labels.to(torch.int64), length)
    return labels.float()


def l_distance(input_vectors: torch.Tensor, ref_vectors: torch.Tensor,
               p: Optional[Union[float, str]] = 2) -> torch.Tensor:
    ns = input_vectors.shape[0]
    nd = input_vectors.shape[-1]
    n_classes = ref_vectors.shape[0]

    dist = (ref_vectors.unsqueeze(0).expand(ns, n_classes, nd) -
            input_vectors.unsqueeze(1).expand(ns, n_classes, nd)).norm(p, dim=-1)
    return dist


def contrastive_distance_loss(input_vectors: torch.Tensor, ref_vectors: torch.Tensor, labels_one_hot: torch.Tensor,
                              p: Optional[Union[float, str]] = 2, temperature: float = 1.) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    dist = l_distance(input_vectors, ref_vectors, p=p)
    sim = -dist / temperature
    total_sim = torch.logsumexp(sim, dim=-1)
    loss = -torch.mean(sim.masked_select(labels_one_hot.bool()) - total_sim)
    pred = torch.argmax(sim, dim=-1)
    return loss, pred


def triplet_margin_loss(input_vectors: torch.Tensor, ref_vectors: torch.Tensor, labels_one_hot: torch.Tensor,
                        p: Optional[Union[float, str]] = 2, base_margin: float = 1) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    margin = base_margin / np.sqrt(ref_vectors.shape[0])
    dist = l_distance(input_vectors, ref_vectors, p=p)
    true_class_mask = labels_one_hot.bool()
    true_class_dist = dist.masked_select(true_class_mask).unsqueeze(1)
    dist_false_labels = dist.masked_fill(true_class_mask, float("Inf"))
    # false_class_with_min_dist_mask = nn.functional.one_hot(
    #     torch.argmin(dist_false_labels, dim=1), labels_one_hot.shape[-1]).bool()
    # false_class_with_min_dist = dist.masked_select(false_class_with_min_dist_mask)
    loss = torch.mean(nn.functional.relu(true_class_dist - dist_false_labels + margin))
    pred = torch.argmin(dist, dim=-1)
    return loss, pred


def get_gradient(model: nn.Module) -> torch.Tensor:
    """
    Get current gradients of a PyTorch model.

    This collects ALL GRADIENTS of the model in a SINGLE VECTOR.
    """
    grad_vec = []
    for param in model.parameters():
        if param.grad is not None:
            grad_vec.append(param.grad.view(-1))
        else:
            # Part of the network might has no grad, fill zero for those terms
            grad_vec.append(param.data.clone().fill_(0).view(-1))

    return torch.cat(grad_vec)


def update_gradient(model: nn.Module, new_grad: torch.Tensor) -> None:
    """
    Overwrite current gradient values in Pytorch model.
    This expects a SINGLE VECTOR containing all corresponding gradients for the model.
    This means that the number of elements of the vector must match the number of gradients in the model.
    """
    ptr = 0
    for param in model.parameters():
        num_params = param.numel()  # number of elements
        if param.grad is not None:
            # replace current param's gradients (in-place) with values from new gradient
            param.grad.copy_(new_grad[ptr:(ptr + num_params)].view_as(param))

        ptr += num_params


def transform_labels_names_to_vector(labels_names: Iterable[str], num_seen_classes: int,
                                     class_names_to_idx: Dict[str, int]) -> torch.Tensor:
    labels = torch.zeros((len(labels_names), num_seen_classes))
    for i in range(len(labels)):
        for j in range(len(labels_names[i])):
            if labels_names[i][j] != NO_LABEL_PLACEHOLDER:
                label_idx = class_names_to_idx[labels_names[i][j]]
                labels[i, label_idx] = 1
    return labels


def copy_freeze(model: nn.Module) -> nn.Module:
    """
    Create a copy of the model, with all the parameters frozer (requires_grad set to False)

    Args:
        model (nn.Module): The model that needs to be copied

    Returns:
        nn.Module: The frozen model
    """
    model_copy = deepcopy(model)
    for param in model_copy.parameters():
        param.requires_grad = False
    return model_copy


def save_model(save_file: str, config: Dict, metadata: Dict, model: Optional['BaseMethod'] = None,
               buffer: Optional['BufferBase'] = None, datasets: Optional[Dict[str, Dataset]] = None,
               **kwargs) -> None:
    """
    Saves the experiment configuration and the state dicts of the model, buffer, datasets, plus any additional data

    Args:
        save_file (str): The checkpointing file path
        config (Dict): The config of the experiment
        metadata (Dict): The metadata of the experiment
        model (Optional[BaseMethod]): The Method object (subclass of BaseMethod) for which the state dict should
            be saved (Default: None)
        buffer (Optional[BufferBase]): The buffer object for which the state dict should be saved (Default: None)
        datasets (Optional[Dict[str, Dataset]]): The different dataset splits for which the state dict should be
            saved (Default: None)
        **kwargs: Any additional key value pairs that need to be saved
    """
    dicts = {}
    dicts['config'] = config
    dicts['metadata'] = metadata
    if model is not None:
        dicts['method_state_dict'] = model.method_state_dict()
    if buffer is not None:
        dicts['buffer_state_dict'] = buffer.buffer_state_dict()
    if datasets is not None:
        dicts['datasets_state_dict'] = {dataset_type: datasets[dataset_type].dataset_state_dict()
                                        for dataset_type in datasets.keys()}
    for key, value in kwargs.items():
        dicts[key] = value
    torch.save(dicts, save_file)


def load_model(checkpoint: Dict[str, Dict], model: Optional['BaseMethod'] = None,
               buffer: Optional['BufferBase'] = None,
               datasets: Optional[Dict[str, Dataset]] = None) -> None:
    """
    Loads the state dicts of the model, buffer and datasets

    Args:
        checkpoint (Dict[str, Dict]): A dictionary of the state dictionaries
        model (Optional[BaseMethod]): The Method object (subclass of BaseMethod) for which the state dict should
            be updated (Default: None)
        buffer (Optional[BufferBase]): The buffer object for which the state dict should be updated (Default: None)
        datasets (Optional[Dict[str, Dataset]]): The different dataset splits for which the state dict should be
            updated (Default: None)
    """
    if model is not None:
        model.load_method_state_dict(checkpoint["method_state_dict"])
    if buffer is not None:
        buffer.load_buffer_state_dict(checkpoint["buffer_state_dict"])
    if datasets is not None:
        for dataset_type in datasets.keys():
            dataset_state_dict = checkpoint['datasets_state_dict'][dataset_type]
            datasets[dataset_type].load_dataset_state_dict(dataset_state_dict)
