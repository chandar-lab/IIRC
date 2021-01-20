import torch.nn as nn
import torch.distributed as dist
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import warnings
from typing import Optional, Union, List, Dict, Callable, Tuple
import math

from iirc.lifelong_dataset.torch_dataset import Dataset
from iirc.definitions import NO_LABEL_PLACEHOLDER
from iirc.utils.utils import print_msg
from lifelong_methods.buffer.buffer import BufferBase
from lifelong_methods.methods.base_method import BaseMethod
from lifelong_methods.utils import SubsetSampler, copy_freeze
from lifelong_methods.models import cosine_linear
from lifelong_methods.models.resnetcifar import ResNetCIFAR
from lifelong_methods.models.resnet import ResNet


class Model(BaseMethod):
    """
    An  implementation of LUCIR from
        Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and Dahua Lin.
        Learning a Unified Classifier Incrementally via Rebalancing.
        CVPR, 2019.
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], config: Dict):
        super(Model, self).__init__(n_cla_per_tsk, class_names_to_idx, config)

        self.sigma = True
        device = next(self.net.parameters()).device
        self.net.model.output_layer = cosine_linear.CosineLinear(in_features=self.latent_dim,
                                                                 out_features=n_cla_per_tsk[0],
                                                                 sigma=self.sigma).to(device)
        self.reset_optimizer_and_scheduler()
        self.old_net = copy_freeze(self.net)  # type: Union[ResNet, ResNetCIFAR]

        self.batch_size = config["batch_size"]

        self.lambda_base = config["lucir_lambda"]
        self.lambda_cur = self.lambda_base
        self.K = 2
        self.margin_1 = config["lucir_margin_1"]
        self.margin_2 = config["lucir_margin_2"]

        # setup losses
        # self.loss_classification = nn.CrossEntropyLoss(reduction="mean")
        self.loss_classification = nn.BCEWithLogitsLoss(reduction="mean")
        self.loss_distill = nn.CosineEmbeddingLoss(reduction="mean")
        # several losses to allow for the use of different margins
        self.loss_mr_1 = nn.MarginRankingLoss(margin=self.margin_1, reduction="mean")
        self.loss_mr_2 = nn.MarginRankingLoss(margin=self.margin_2, reduction="mean")

        self.method_variables.extend(["lambda_base", "lambda_cur", "K", "margin_1", "margin_2", "sigma"])

    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This is where anything model specific needs to be done before the state_dicts are loaded.
        This method replaces the output layer of the vanilla resnet with the cosine layer, and change the trainable
        parameters.

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        assert "method_variables" in state_dicts.keys()
        method_variables = state_dicts['method_variables']
        cur_task_id = method_variables["cur_task_id"]
        n_cla_per_tsk = method_variables["n_cla_per_tsk"]
        num_old_classes = int(sum(n_cla_per_tsk[: cur_task_id]))
        num_new_classes = n_cla_per_tsk[cur_task_id]
        device = next(self.net.parameters()).device
        if cur_task_id > 0:
            self.net.model.output_layer = cosine_linear.SplitCosineLinear(in_features=self.latent_dim,
                                                                          out_features1=num_old_classes,
                                                                          out_features2=num_new_classes,
                                                                          sigma=self.sigma).to(device)
            trainable_parameters = [param for name, param in self.net.named_parameters() if
                                    "output_layer.fc1" not in name]
            self.reset_optimizer_and_scheduler(trainable_parameters)
            if cur_task_id > 1:
                out_features1 = int(sum(n_cla_per_tsk[: cur_task_id - 1]))
                out_features2 = n_cla_per_tsk[cur_task_id - 1]
                self.old_net.model.output_layer = cosine_linear.SplitCosineLinear(in_features=self.latent_dim,
                                                                                  out_features1=out_features1,
                                                                                  out_features2=out_features2,
                                                                                  sigma=self.sigma).to(device)

    def _prepare_model_for_new_task(self, task_data: Dataset, dist_args: Optional[dict] = None,
                                    **kwargs) -> None:
        """
        A method specific function that takes place before the starting epoch of each new task (runs from the
        prepare_model_for_task function).
        It copies the old network and freezes it's gradients.
        It also extends the output layer, imprints weights for those extended nodes, and change the trainable parameters

        Args:
            task_data (Dataset): The new task dataset
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
        """
        self.old_net = copy_freeze(self.net)
        self.old_net.eval()

        cur_task_id = self.cur_task_id
        num_old_classes = int(sum(self.n_cla_per_tsk[: cur_task_id]))
        num_new_classes = self.n_cla_per_tsk[cur_task_id]
        device = next(self.net.parameters()).device

        # Extend last layer
        if cur_task_id > 0:
            output_layer = cosine_linear.SplitCosineLinear(in_features=self.latent_dim,
                                                           out_features1=num_old_classes,
                                                           out_features2=num_new_classes,
                                                           sigma=self.sigma).to(device)
            if cur_task_id == 1:
                output_layer.fc1.weight.data = self.net.model.output_layer.weight.data
            else:
                out_features1 = self.net.model.output_layer.fc1.out_features
                output_layer.fc1.weight.data[:out_features1] = self.net.model.output_layer.fc1.weight.data
                output_layer.fc1.weight.data[out_features1:] = self.net.model.output_layer.fc2.weight.data
            output_layer.sigma.data = self.net.model.output_layer.sigma.data
            self.net.model.output_layer = output_layer
            self.lambda_cur = self.lambda_base * math.sqrt(num_old_classes * 1.0 / num_new_classes)
            print_msg(f"Lambda for less forget is set to {self.lambda_cur}")
        elif cur_task_id != 0:
            raise ValueError("task id cannot be negative")

        # Imprint weights
        with task_data.disable_augmentations():
            if cur_task_id > 0:
                print_msg("Imprinting weights")
                self.net = self._imprint_weights(task_data, self.net, dist_args)

        # Fix parameters of FC1 for less forget and reset optimizer/scheduler
        if cur_task_id > 0:
            trainable_parameters = [param for name, param in self.net.named_parameters() if
                                    "output_layer.fc1" not in name]
        else:
            trainable_parameters = self.net.parameters()
        self.reset_optimizer_and_scheduler(trainable_parameters)

    def _imprint_weights(self, task_data: Dataset, model: Union[ResNet, ResNetCIFAR],
                         dist_args: Optional[dict] = None) -> Union[ResNet, ResNetCIFAR]:
        distributed = dist_args is not None
        if distributed:
            device = torch.device(f"cuda:{dist_args['gpu']}")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class_names = task_data.cur_task
        class_names_2_idx = self.class_names_to_idx
        model.eval()
        num_old_classes = model.model.output_layer.fc1.out_features
        old_weights_norm = model.model.output_layer.fc1.weight.data.norm(dim=1, keepdim=True)
        average_old_weights_norm = torch.mean(old_weights_norm, dim=0)
        new_weights = torch.zeros_like(model.model.output_layer.fc2.weight.data)
        for cla in class_names:
            cla_id = class_names_2_idx[cla]
            if cla_id < num_old_classes:
                continue
            num_samples = 1000
            class_indices = task_data.get_image_indices_by_cla(cla, num_samples=num_samples, shuffle=False)
            if distributed:  # make sure all the gpus use the same random indices
                class_data_indices_to_broadcast = torch.from_numpy(class_indices).to(device)
                torch.distributed.broadcast(class_data_indices_to_broadcast, 0)
                class_indices = class_data_indices_to_broadcast.cpu().numpy()
            sampler = SubsetSampler(class_indices)
            class_loader = DataLoader(task_data, batch_size=self.batch_size, sampler=sampler)
            normalized_latent_feat = []
            with torch.no_grad():
                for minibatch in class_loader:
                    inputs = minibatch[0].to(device)
                    output, latent_features = model(inputs)
                    latent_features = latent_features.detach()
                    latent_features = F.normalize(latent_features, p=2, dim=-1)
                    normalized_latent_feat.append(latent_features)
                normalized_latent_feat = torch.cat(normalized_latent_feat, dim=0)
                mean_latent_feat = torch.mean(normalized_latent_feat, dim=0)
                normalized_mean_latent = F.normalize(mean_latent_feat, p=2, dim=0)
                new_weights[cla_id - num_old_classes] = normalized_mean_latent * average_old_weights_norm
        model.model.output_layer.fc2.weight.data = new_weights
        return model

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
        device = x.device
        num_seen_classes = len(self.seen_classes)
        offset_1, offset_2 = self._compute_offsets(self.cur_task_id)
        target = y
        assert y.shape[1] == offset_2 == num_seen_classes
        output, latent_feat = self.forward_net(x)
        assert output.shape[1] == num_seen_classes

        loss_1 = self.loss_classification(output, target)
        if self.cur_task_id > 0:
            self.old_net.eval()
            _, old_latent_feat = self.old_net(x)
            old_latent_feat = old_latent_feat.detach()
            loss_2 = self.loss_distill(latent_feat.clone(), old_latent_feat,
                                       torch.ones(x.shape[0]).to(device)) * self.lambda_cur
            loss_3 = torch.zeros_like(loss_1)

            if in_buffer is not None and torch.sum(in_buffer) > 0:
                buffer_target = y[in_buffer].bool()
                buffer_output = output[in_buffer]
                sigma = self.net.model.output_layer.sigma
                if sigma is not None:
                    buffer_output /= sigma.data
                assert buffer_target.shape[0] == torch.sum(in_buffer)  ####
                # target_scores = buffer_output.masked_select(buffer_target).view(-1, 1).repeat(1, self.K).reshape(-1)
                target_scores = buffer_output.masked_select(buffer_target).reshape(-1)
                new_classes_scores = buffer_output[:, offset_1:]
                topk_scores, _ = new_classes_scores.topk(self.K, dim=-1)
                topk_scores_1 = topk_scores[:, 0].reshape(-1)
                topk_scores_2 = topk_scores[:, 1:].reshape(-1)
                loss_3 = self.loss_mr_1(target_scores.reshape(-1), topk_scores_1, torch.ones_like(topk_scores_1))
                loss_3 += self.loss_mr_2(target_scores.view(-1, 1).repeat(1, self.K - 1).reshape(-1), topk_scores_2,
                                         torch.ones_like(topk_scores_2)) * (self.K - 1)
                loss_3 /= self.K
        else:
            loss_2 = torch.zeros_like(loss_1)
            loss_3 = torch.zeros_like(loss_1)

        if train:
            loss = loss_1 + loss_2 + loss_3
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        else:
            loss = loss_1

        predictions = output.ge(0.0)

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
        assert output.shape[1] == num_seen_classes
        predictions = output.ge(0.0)
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
        """remove extra exemplars from the buffer"""
        for label in self.seen_classes:
            if len(self.mem_class_x[label]) > self.n_mems_per_cla:
                n = len(self.mem_class_x[label]) - self.n_mems_per_cla
                self.remove_samples(label, n)

    def _construct_exemplar_set(self, task_data: Dataset, dist_args: Optional[dict] = None,
                                model: torch.nn.Module = None, batch_size=1, **kwargs) -> None:
        """
        Update the buffer with the new task samples using herding

        Args:
            task_data (Dataset): The new task data
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
            model (BaseMethod): The current method object to calculate the latent variables
            batch_size (int): The minibatch size
        """
        distributed = dist_args is not None
        if distributed:
            device = torch.device(f"cuda:{dist_args['gpu']}")
            rank = dist_args['rank']
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rank = 0
        new_class_labels = task_data.cur_task
        model.eval()

        print_msg(f"Adding buffer samples")  ####
        with task_data.disable_augmentations():  # disable augmentations then enable them (if they were already enabled)
            with torch.no_grad():
                for class_label in new_class_labels:
                    class_data_indices = task_data.get_image_indices_by_cla(class_label, self.max_mems_pool_size)
                    if distributed:
                        device = torch.device(f"cuda:{dist_args['gpu']}")
                        class_data_indices_to_broadcast = torch.from_numpy(class_data_indices).to(device)
                        torch.distributed.broadcast(class_data_indices_to_broadcast, 0)
                        class_data_indices = class_data_indices_to_broadcast.cpu().numpy()
                    sampler = SubsetSampler(class_data_indices)
                    class_loader = DataLoader(task_data, batch_size=batch_size, sampler=sampler)
                    latent_vectors = []
                    for minibatch in class_loader:
                        images = minibatch[0].to(device)
                        output, out_latent = model.forward_net(images)
                        out_latent = out_latent.detach()
                        out_latent = F.normalize(out_latent, p=2, dim=-1)
                        latent_vectors.append(out_latent)
                    latent_vectors = torch.cat(latent_vectors, dim=0)
                    class_mean = torch.mean(latent_vectors, dim=0)

                    chosen_exemplars_ind = []
                    exemplars_mean = torch.zeros_like(class_mean)
                    while len(chosen_exemplars_ind) < min(self.n_mems_per_cla, len(class_data_indices)):
                        potential_exemplars_mean = (exemplars_mean.unsqueeze(0) * len(
                            chosen_exemplars_ind) + latent_vectors) \
                                                   / (len(chosen_exemplars_ind) + 1)
                        distance = (class_mean.unsqueeze(0) - potential_exemplars_mean).norm(dim=-1)
                        shuffled_index = torch.argmin(distance).item()
                        exemplars_mean = potential_exemplars_mean[shuffled_index, :].clone()
                        exemplar_index = class_data_indices[shuffled_index]
                        chosen_exemplars_ind.append(exemplar_index)
                        latent_vectors[shuffled_index, :] = float("inf")

                    for image_index in chosen_exemplars_ind:
                        image, label1, label2 = task_data.get_item(image_index)
                        if label2 != NO_LABEL_PLACEHOLDER:
                            warnings.warn(f"Sample is being added to the buffer with labels {label1} and {label2}")
                        self.add_sample(class_label, image, (label1, label2), rank=rank)
