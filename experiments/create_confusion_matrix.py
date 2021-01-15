import torch
import argparse
import numpy as np
import random
import torchvision
import json
import os
import importlib
import torch.utils.data as data
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional

from iirc.datasets_loader import get_lifelong_datasets
from lifelong_methods.methods.base_method import BaseMethod
from lifelong_methods.utils import SubsetSampler


def get_transforms(dataset_name):
    transforms = None
    if "cifar100" in dataset_name:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
    elif "imagenet" in dataset_name:
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    return transforms


def get_model(args, device, task_id=0):
    if args.models_dir == "None":
        model = None
    else:
        checkpoint = torch.load(os.path.join(args.models_dir, f"task_{task_id}_model"))
        metadata = checkpoint['metadata']
        config = checkpoint['config']
        assert os.path.isdir('experiments/methods')
        method = importlib.import_module('experiments.methods.' + config["method"])
        model = method.Model(metadata["n_cla_per_tsk"], metadata["class_names_to_idx"],
                             config)  # type: Optional[BaseMethod]
        model.to(device)
        method_state_dict = checkpoint["method_state_dict"]
        model._load_method_state_dict(method_state_dict)
        model.load_state_dict(method_state_dict['model_state_dict'])
        for key, value in method_state_dict['method_variables'].items():
            model.__dict__[key] = value
        model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="iirc_cifar100",
                        choices=["incremental_cifar100", "iirc_cifar100", "iirc_imagenet_full", "iirc_imagenet_lite"])
    parser.add_argument('--dataset_path', type=str, default="./data")
    parser.add_argument('--metadata_root', type=str, default="./iirc/metadata")
    parser.add_argument('--results_path', type=str, default="./figures/confusion_matrices/ground_truth")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--models_dir', type=str, default="None")
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--tasks_configuration_id', type=int, default=0)
    parser.add_argument('--joint', action="store_true")
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(args.metadata_root, "iirc_cifar100_hierarchy.json"), "r") as f:
        class_hierarchy = json.load(f)

    superclass_to_subclasses = class_hierarchy['super_classes']
    subclasses_to_superclass = {subclass: superclass for superclass, subclasses in superclass_to_subclasses.items()
                                for subclass in subclasses}

    essential_transforms_fn = get_transforms(args.dataset)
    lifelong_datasets, tasks, class_names_to_idx = \
        get_lifelong_datasets(args.dataset, dataset_root=args.dataset_path,
                              tasks_configuration_id=args.tasks_configuration_id,
                              essential_transforms_fn=essential_transforms_fn, cache_images=False, joint=args.joint)

    dataset = lifelong_datasets["posttask_valid"]
    model = None

    tasks_id_to_show = [0, 1, 5, 10]
    num_classes_per_task = [10, 5, 5, 5]
    confusions = [np.zeros((sum(num_classes_per_task), sum(num_classes_per_task))) for _ in
                  range(len(tasks_id_to_show))]
    total_classes_to_show = []
    for task_id, num_classes in zip(tasks_id_to_show, num_classes_per_task):
        total_classes_to_show.extend(tasks[task_id][:num_classes])
    class_names_to_idx = {cla: idx for idx, cla in enumerate(total_classes_to_show)}
    classes_shown = []

    for i, (task_id, num_classes) in enumerate(zip(tasks_id_to_show, num_classes_per_task)):
        confusion = confusions[i]
        task_classes = tasks[task_id][:num_classes]
        classes_shown.extend(task_classes)
        dataset.load_tasks_up_to(task_id)
        model = get_model(args, device, task_id)
        for class_ in classes_shown:
            class_data_indices = dataset.get_image_indices_by_cla(class_)
            class_idx = class_names_to_idx[class_]
            if model is None:
                confusion[class_idx, class_idx] = 1.0
                if class_ in subclasses_to_superclass.keys():
                    super_cla = subclasses_to_superclass[class_]
                    if super_cla in classes_shown:
                        super_cla_idx = class_names_to_idx[super_cla]
                        confusion[class_idx, super_cla_idx] = 1.0
                if class_ in superclass_to_subclasses.keys():
                    for sub_cla in superclass_to_subclasses[class_]:
                        if sub_cla in classes_shown:
                            sub_cla_idx = class_names_to_idx[sub_cla]
                            confusion[class_idx, sub_cla_idx] = 1.0 / len(superclass_to_subclasses[class_])
            else:
                sampler = SubsetSampler(class_data_indices)
                class_loader = data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
                model_class_names_to_idx = model.class_names_to_idx
                relevant_idx = np.array([model_class_names_to_idx[cla] for cla in classes_shown])
                total_class_predictions = torch.zeros(len(classes_shown)).to(device)
                for minibatch in class_loader:
                    images = minibatch[0].to(device)
                    predictions = model(images)
                    predictions = predictions[:, relevant_idx]
                    total_class_predictions += predictions.sum(dim=0)
                total_class_predictions /= (1.0 * len(class_data_indices))
                confusion[class_idx, :len(classes_shown)] = total_class_predictions.cpu().numpy()

    os.makedirs(args.results_path, exist_ok=True)
    for task_id, confusion in zip(tasks_id_to_show, confusions):
        confusion_df = pd.DataFrame(confusion, total_classes_to_show,
                                    total_classes_to_show)
        matplotlib.use('Agg')
        sn.heatmap(confusion_df * 100.0, vmin=0, vmax=100, annot=False, yticklabels=False, xticklabels=False,
                   fmt='.1f')  # , annot_kws={"size": 8}
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(15, 12)
        fig.tight_layout()
        plt.savefig(os.path.join(args.results_path, f"task_{task_id}_confusion.png"))
        plt.close()
        confusion_df = pd.DataFrame(confusion, total_classes_to_show,
                                    total_classes_to_show)
        matplotlib.use('Agg')
        sn.heatmap(confusion_df * 100.0, vmin=0, vmax=100, annot=True, fmt='.1f')  # , annot_kws={"size": 8}
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(15, 12)
        fig.tight_layout()
        plt.savefig(os.path.join(args.results_path, f"task_{task_id}_confusion_annot.png"))
        plt.close()
