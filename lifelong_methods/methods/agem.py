import torch.nn as nn
import torch.distributed as dist
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import warnings
from typing import Optional, Union, List, Dict, Callable, Tuple

from iirc.lifelong_dataset.torch_dataset import Dataset
from iirc.definitions import NO_LABEL_PLACEHOLDER
from lifelong_methods.buffer.buffer import BufferBase
from lifelong_methods.methods.base_method import BaseMethod
from lifelong_methods.utils import transform_labels_names_to_vector, get_gradient, update_gradient


class Model(BaseMethod):
    """
    An  implementation of A-GEM from
        Arslan Chaudhry, Marc’Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny.
        Efficient Lifelong Learning with A-GEM.
        ICLR, 2019.
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], config: Dict):
        super(Model, self).__init__(n_cla_per_tsk, class_names_to_idx, config)
        # setup losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.device = config["device"]
        self.memory_batch_size = config["batch_size"]
        self.memory_loader = None

        self.method_variables.extend(["memory_batch_size"])

    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This is where anything model specific needs to be done before the state_dicts are loaded

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        pass

    def _prepare_model_for_new_task(self, buffer: Optional[BufferBase] = None, dist_args: Optional[Dict] = None,
                                    num_workers: int = 2, **kwargs) -> None:
        """
        A method specific function that takes place before the starting epoch of each new task (runs from the
        prepare_model_for_task function).
        It checks if the buffer is not empty, and creates a dataloader for it, which is used later in getting the
            gradients of the loss on the buffer samples.

        Args:
            buffer (Optional[BufferBase]): The replay buffer (default: None)
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
            num_workers (int): The number of workers to use with the dataloader of the replay buffer (default: 2)
        """
        if dist_args is not None:
            raise NotImplementedError
        if buffer is not None and len(buffer) > 0:
            self.memory_loader = DataLoader(buffer, batch_size=self.memory_batch_size, num_workers=num_workers,
                                            shuffle=True)

    def _get_buffer_gradient(self) -> torch.Tensor:
        device = self.device
        buffer_gradient = None  # type: Optional[torch.Tensor]
        if self.memory_loader is not None:
            buffer_batch = next(iter(self.memory_loader))
            samples = buffer_batch[0].to(device)
            labels_names = list(zip(buffer_batch[1], buffer_batch[2]))
            labels = transform_labels_names_to_vector(
                labels_names, len(self.seen_classes), self.class_names_to_idx
            )
            labels = labels.to(device)
            offset_1, offset_2 = self._compute_offsets(self.cur_task_id)
            output, _ = self.forward_net(samples)
            output = output[:, :offset_1]
            labels = labels[:, :offset_1]
            loss = self.bce(output / self.temperature, labels)
            self.opt.zero_grad()
            loss.backward()
            buffer_gradient = get_gradient(self.net)
        return buffer_gradient

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
        if in_buffer is not None:
            assert torch.sum(in_buffer).item() == 0
        num_seen_classes = len(self.seen_classes)
        offset_1, offset_2 = self._compute_offsets(self.cur_task_id)
        target = y
        assert target.shape[1] == offset_2
        output, _ = self.forward_net(x)
        output = output[:, :offset_2]

        buffer_gradient = None  # type: Optional[torch.Tensor]
        if train and self.memory_loader is not None:
            buffer_gradient = self._get_buffer_gradient()

        loss = self.bce(output[:, offset_1:] / self.temperature, target[:, offset_1:])

        # TODO weigh the buffer loss by the self.memory_strength before getting the loss mean (use in_buffer)
        if train:
            self.opt.zero_grad()
            loss.backward()
            if buffer_gradient is not None:
                cur_gradient = get_gradient(self.net)
                dotp = torch.dot(cur_gradient, buffer_gradient)
                if dotp < 0:
                    # efficient gradient projection
                    ref_mag = torch.dot(buffer_gradient, buffer_gradient)
                    new_grad = cur_gradient - ((dotp / ref_mag) * buffer_gradient)
                    update_gradient(self.net, new_grad)
            self.opt.step()

        predictions = output > 0.0
        return predictions, loss.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The method used during inference, returns a tensor of model predictions

        Args:
            x (torch.Tensor): The batch of images

        Returns:
            torch.Tensor: a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
        """
        num_seen_classes = len(self.seen_classes)
        output, _ = self.forward_net(x)
        output = output[:, :num_seen_classes]
        predictions = output > 0.0
        return predictions

    def _consolidate_epoch_knowledge(self, **kwargs) -> None:
        """
        A method specific function that takes place after training on each epoch (runs from the
        consolidate_epoch_knowledge function)
        """
        pass

    def consolidate_task_knowledge(self, **kwargs) -> None:
        """Takes place after training on each task"""
        pass


class Buffer(BufferBase):
    def __init__(self,
                 config: Dict,
                 buffer_dir: Optional[str] = None,
                 map_size: int = 1e9,
                 essential_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                 augmentation_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        super(Buffer, self).__init__(config, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)

    def _reduce_exemplar_set(self, **kwargs) -> None:
        """
        remove extra exemplars from the buffer
        """
        for label in self.seen_classes:
            if len(self.mem_class_x[label]) > self.n_mems_per_cla:
                n = len(self.mem_class_x[label]) - self.n_mems_per_cla
                self.remove_samples(label, n)

    def _construct_exemplar_set(self, task_data: Dataset, dist_args: Optional[Dict] = None, **kwargs) -> None:
        """
        update the buffer with the new task exemplars, chosen randomly for each class.

        Args:
            new_task_data (Dataset): The new task data
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
        """
        distributed = dist_args is not None
        if distributed:
            rank = dist_args['rank']
        else:
            rank = 0
        new_class_labels = task_data.cur_task

        for class_label in new_class_labels:
            num_images_to_add = min(self.n_mems_per_cla, self.max_mems_pool_size)
            class_images_indices = task_data.get_image_indices_by_cla(class_label, num_images_to_add)
            if distributed:
                device = torch.device(f"cuda:{dist_args['gpu']}")
                class_images_indices_to_broadcast = torch.from_numpy(class_images_indices).to(device)
                torch.distributed.broadcast(class_images_indices_to_broadcast, 0)
                class_images_indices = class_images_indices_to_broadcast.cpu().numpy()

            for image_index in class_images_indices:
                image, label1, label2 = task_data.get_item(image_index)
                if label2 != NO_LABEL_PLACEHOLDER:
                    warnings.warn(f"Sample is being added to the buffer with labels {label1} and {label2}")
                self.add_sample(class_label, image, (label1, label2), rank=rank)
