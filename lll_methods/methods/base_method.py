from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, List, Dict, Iterator, Tuple

from lll_methods.models.resnetcifar import ResNetCIFAR
from lll_methods.models.resnet import ResNet
from lll_methods.utils import get_optimizer
from iirc.utils.utils import print_msg
from iirc.lll_dataset.torch_lll_dataset import TorchLLLDataset


class LLLMethodBase(ABC, nn.Module):
    """
    A base model for all the lifelong learning methods to inherit, which contains all common functionality

    Args:
        n_cla_per_tsk (Union[np.ndarray, List[int]]): An integer numpy array including the number of classes per each task.
        class_names_to_idx (Dict[str, int]): The index of each class name
        config (Dict): A dictionary that has the following key value pairs:
            temperature (float): the temperature to divide the logits by
            memory_strength (float): The weight to add for the samples from the buffer when computing the loss
                (not implemented yet)
            n_layers (int): The number of layers for the network used (not all values are allowed depending on the
                architecture)
            dataset (str): The name of the dataset (for ex: iirc_cifar100)
            optimizer (str): The type of optimizer ("momentum" or "adam")
            lr (float): The initial learning rate
            lr_schedule (Optional[list[int]]): The epochs for which the learning rate changes
            lr_gamma (float): The multiplier multiplied by the learning rate at the epochs specified in lr_schedule
            reduce_lr_on_plateau (bool): reduce learning rate on plateau
            weight_decay (float): the weight decay multiplier
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], config: Dict):
        super(LLLMethodBase, self).__init__()
        self.n_cla_per_tsk = n_cla_per_tsk
        self.class_names_to_idx = class_names_to_idx
        self.num_classes = int(sum(self.n_cla_per_tsk))
        self.cur_task_id = 0  # The current training task id
        self.temperature = config["temperature"]
        self.memory_strength = config["memory_strength"]
        self.n_layers = config["n_layers"]
        self.seen_classes = []

        # setup network
        if 'cifar' in config["dataset"]:
            self.net = ResNetCIFAR(num_classes=self.num_classes, num_layers=self.n_layers, relu_last_hidden=False)
        elif 'imagenet' in config["dataset"]:
            self.net = ResNet(num_classes=self.num_classes, num_layers=self.n_layers)
        else:
            raise ValueError(f"Unsupported dataset {config['dataset']}")
        self.latent_dim = self.net.model.output_layer.in_features

        # setup optimizer
        self.optimizer_type = config["optimizer"]
        self.lr = config["lr"]
        self.lr_gamma = config["lr_gamma"]
        self.lr_schedule = config["lr_schedule"]
        self.reduce_lr_on_plateau = config["reduce_lr_on_plateau"]
        self.weight_decay = config["weight_decay"]
        self.opt, self.scheduler = get_optimizer(
            model_parameters=self.net.parameters(), optimizer_type=self.optimizer_type, lr=self.lr,
            lr_gamma=self.lr_gamma, lr_schedule=self.lr_schedule, reduce_lr_on_plateau=self.reduce_lr_on_plateau,
            weight_decay=self.weight_decay
        )

        # The model variables that are not in the state_dicts of the model and that need to be saved
        self.method_variables = ['n_cla_per_tsk', 'num_classes', 'cur_task_id', 'temperature', 'memory_strength',
                                 'n_layers', 'seen_classes', 'latent_dim', 'optimizer_type', 'lr', 'lr_gamma',
                                 'lr_schedule', 'weight_decay']
        for variable in self.method_variables:
            assert variable in self.__dict__.keys()

    def method_state_dict(self) -> Dict[str, Dict]:
        """
        This function returns a dict that contains the state dictionaries of this method (including the model, the
            optimizer, the scheduler, as well as the values of the variables whose names are inside the
            self.method_variables), so that they can be used for checkpointing.

        Returns:
            Dict: a dictionary with the state dictionaries of this method, the optimizer, the scheduler, and the values
            of the variables whose names are inside the self.method_variables
        """
        state_dicts = {}
        state_dicts['model_state_dict'] = self.state_dict()
        state_dicts['optimizer_state_dict'] = self.opt.state_dict()
        state_dicts['scheduler_state_dict'] = self.scheduler.state_dict()
        state_dicts['method_variables'] = {key: self.__dict__[key] for key in self.method_variables}
        return state_dicts

    def load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This function loads the state dicts of the various parts of this method (along with the variables in
            self.method_variables)

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        self._load_method_state_dict(state_dicts)
        keys = {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'method_variables'}
        assert keys.issubset(state_dicts.keys())
        assert set(self.method_variables) == set(state_dicts['method_variables'].keys())
        self.load_state_dict(state_dicts['model_state_dict'])
        self.opt.load_state_dict(state_dicts['optimizer_state_dict'])
        self.scheduler.load_state_dict(state_dicts['scheduler_state_dict'])
        for key, value in state_dicts['method_variables'].items():
            self.__dict__[key] = value

    @abstractmethod
    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This is where anything model specific needs to be done before the state_dicts are loaded
        This function needs to be defined in the inheriting method class

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        pass

    def reset_optimizer_and_scheduler(self, optimizable_parameters: Optional[
        Iterator[nn.parameter.Parameter]] = None) -> None:
        """
        Reset the optimizer and scheduler after a task is done (with the option to specify which parameters to optimize

        Args:
            optimizable_parameters (Optional[Iterator[nn.parameter.Parameter]]: specify the parameters that should be
                optimized, in case some parameters needs to be frozen (default: None)
        """
        print_msg(f"resetting scheduler and optimizer, learning rate = {self.lr}")
        if optimizable_parameters is None:
            optimizable_parameters = self.net.parameters()
        self.opt, self.scheduler = get_optimizer(
            model_parameters=optimizable_parameters, optimizer_type=self.optimizer_type, lr=self.lr,
            lr_gamma=self.lr_gamma, lr_schedule=self.lr_schedule, reduce_lr_on_plateau=self.reduce_lr_on_plateau,
            weight_decay=self.weight_decay
        )

    def get_last_lr(self) -> List[float]:
        """Get the current learning rate"""
        lr = [group['lr'] for group in self.opt.param_groups]
        return lr

    def step_scheduler(self, val_metric: Optional = None) -> None:
        """
        Take a step with the scheduler (should be called after each epoch)

        Args:
            val_metric (Optional): a metric to compare in case of reducing the learning rate on plateau (default: None)
        """
        cur_lr = self.get_last_lr()
        if self.reduce_lr_on_plateau:
            assert val_metric is not None
            self.scheduler.step(val_metric)
        else:
            self.scheduler.step()
        new_lr = self.get_last_lr()
        if cur_lr != new_lr:
            print_msg(f"learning rate changes to {new_lr}")

    def _compute_offsets(self, task) -> Tuple[int, int]:
        offset1 = int(sum(self.n_cla_per_tsk[:task]))
        offset2 = int(sum(self.n_cla_per_tsk[:task + 1]))
        return offset1, offset2

    def forward_net(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        an alias for self.net(x)

        Args:
            x (torch.Tensor): The batch of images

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            output (torch.Tensor): The network output of shape (minibatch size x output size)
            latent (torch.Tensor): The network latent variable of shape (minibatch size x last hidden size)
        """
        return self.net(x)

    def prepare_model_for_new_task(self, task_data: Optional[TorchLLLDataset] = None, dist_args: Optional[dict] = None,
                                   **kwargs) -> None:
        """
        Takes place before the starting epoch of each new task.

        The shared functionality among the methods is that the seen classes are updated and the optimizer and scheduler
        are reset. (see _prepare_model_for_new_task for method specific functionality)

        Args:
            task_data (Optional[TorchLLLDataset]): The new task data (default: None)
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
            **kwargs: arguments that are method specific
        """
        self.seen_classes = list(set(self.seen_classes) | set(task_data.cur_task))
        self.reset_optimizer_and_scheduler()
        if task_data.cur_task_id > self.cur_task_id:
            self.cur_task_id = task_data.cur_task_id
        self._prepare_model_for_new_task(task_data=task_data, dist_args=dist_args, **kwargs)

    @abstractmethod
    def _prepare_model_for_new_task(self, **kwargs) -> None:
        """
        A method specific function that takes place before the starting epoch of each new task (runs from the
            prepare_model_for_task function)
        This function needs to be defined in the inheriting method class
        """
        pass

    def consolidate_epoch_knowledge(self, val_metric=None, **kwargs) -> None:
        """
        Takes place after training on each epoch

        The shared functionality among the methods is that the scheduler takes a step. (see _consolidate_epoch_knowledge
        for method specific functionality)

        Args:
            val_metric (Optional): a metric to compare in case of reducing the learning rate on plateau (default: None)
            **kwargs: arguments that are method specific
        """
        self.step_scheduler(val_metric)
        self._consolidate_epoch_knowledge(**kwargs)

    @abstractmethod
    def _consolidate_epoch_knowledge(self, **kwargs) -> None:
        """
        A method specific function that takes place after training on each epoch (runs from the
        consolidate_epoch_knowledge function)
        This function needs to be defined in the inheriting method class
        """
        pass

    @abstractmethod
    def observe(self, x: torch.Tensor, y: torch.Tensor, in_buffer: Optional[torch.Tensor] = None,
                train: bool = True) -> Tuple[torch.Tensor, float]:
        """
        The method used for training and validation, returns a tensor of model predictions and the loss
        This function needs to be defined in the inheriting method class

        Args:
            x (torch.Tensor): The batch of images
            y (torch.Tensor): A 2-d batch indicator tensor of shape (number of samples x number of classes)
            in_buffer (Optional[torch.Tensor]): A 1-d boolean tensor which indicates which sample is from the buffer.
            train (bool): Whether this is training or validation/test

        Returns:
            Tuple[torch.Tensor, float]:
            predictions (torch.Tensor) : a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
            loss (float): the value of the loss
        """
        pass

    @abstractmethod
    def consolidate_task_knowledge(self, **kwargs) -> None:
        """
        Takes place after training each task
        This function needs to be defined in the inheriting method class

        Args:
            **kwargs: arguments that are method specific
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The method used during inference, returns a tensor of model predictions
        This function needs to be defined in the inheriting method class

        Args:
            x (torch.Tensor): The batch of images

        Returns:
            torch.Tensor: a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
        """
        pass
