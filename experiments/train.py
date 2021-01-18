import os
import time
import torch
import torch.utils.data as data
import torch.distributed as dist
from copy import deepcopy

from iirc.utils.utils import print_msg
import lifelong_methods.utils
from lifelong_methods.buffer.buffer import TaskDataMergedWithBuffer
from lifelong_methods.utils import transform_labels_names_to_vector
from experiments import utils, metrics


def epoch_train(model, dataloader, config, metadata, gpu=None, rank=0):
    train_loss = 0
    train_metrics = {'jaccard_sim': 0., 'modified_jaccard': 0., 'strict_acc': 0., 'recall': 0.}
    data_len = 0
    class_names_to_idx = metadata["class_names_to_idx"]
    num_seen_classes = len(model.seen_classes)
    model.net.train()

    minibatch_i = 0
    for minibatch in dataloader:
        labels_names = list(zip(minibatch[1], minibatch[2]))
        labels = transform_labels_names_to_vector(
            labels_names, num_seen_classes, class_names_to_idx
        )

        if gpu is None:
            images = minibatch[0].to(config["device"], non_blocking=True)
            labels = labels.to(config["device"], non_blocking=True)
        else:
            images = minibatch[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            labels = labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)

        if len(minibatch) > 3:
            if gpu is None:
                in_buffer = minibatch[3].to(config["device"], non_blocking=True)
            else:
                in_buffer = minibatch[3].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
        else:
            in_buffer = None

        predictions, loss = model.observe(images, labels, in_buffer, train=True)
        labels = labels.bool()
        train_loss += loss * images.shape[0]
        train_metrics['jaccard_sim'] += metrics.jaccard_sim(predictions, labels) * images.shape[0]
        train_metrics['modified_jaccard'] += metrics.modified_jaccard_sim(predictions, labels) * images.shape[0]
        train_metrics['strict_acc'] += metrics.strict_accuracy(predictions, labels) * images.shape[0]
        train_metrics['recall'] += metrics.recall(predictions, labels) * images.shape[0]
        data_len += images.shape[0]

        if minibatch_i == 0:
            print_msg(
                f"rank {rank}, max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB, "
                f"current memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB\n")
        minibatch_i += 1

    train_loss /= data_len
    train_metrics['jaccard_sim'] /= data_len
    train_metrics['modified_jaccard'] /= data_len
    train_metrics['strict_acc'] /= data_len
    train_metrics['recall'] /= data_len
    return train_loss, train_metrics


def evaluate(model, dataloader, config, metadata, test_mode=False, gpu=None):
    valid_loss = 0
    valid_metrics = {'jaccard_sim': 0., 'modified_jaccard': 0., 'strict_acc': 0., 'recall': 0.}
    data_len = 0
    class_names_to_idx = metadata["class_names_to_idx"]
    num_seen_classes = len(model.seen_classes)
    model.net.eval()

    with torch.no_grad():
        for minibatch in dataloader:
            labels_names = list(zip(minibatch[1], minibatch[2]))
            labels = transform_labels_names_to_vector(
                labels_names, num_seen_classes, class_names_to_idx
            )

            if gpu is None:
                images = minibatch[0].to(config["device"], non_blocking=True)
                labels = labels.to(config["device"], non_blocking=True)
            else:
                images = minibatch[0].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
                labels = labels.to(torch.device(f"cuda:{gpu}"), non_blocking=True)

            if len(minibatch) > 3:
                if gpu is None:
                    in_buffer = minibatch[3].to(config["device"], non_blocking=True)
                else:
                    in_buffer = minibatch[3].to(torch.device(f"cuda:{gpu}"), non_blocking=True)
            else:
                in_buffer = None

            if not test_mode:
                predictions, loss = model.observe(images, labels, in_buffer, train=False)
                valid_loss += loss * images.shape[0]
            else:
                predictions = model(images)

            labels = labels.bool()
            valid_metrics['jaccard_sim'] += metrics.jaccard_sim(predictions, labels) * images.shape[0]
            valid_metrics['modified_jaccard'] += metrics.modified_jaccard_sim(predictions, labels) * images.shape[0]
            valid_metrics['strict_acc'] += metrics.strict_accuracy(predictions, labels) * images.shape[0]
            valid_metrics['recall'] += metrics.recall(predictions, labels) * images.shape[0]
            data_len += images.shape[0]
        valid_loss /= data_len
        valid_metrics['jaccard_sim'] /= data_len
        valid_metrics['modified_jaccard'] /= data_len
        valid_metrics['strict_acc'] /= data_len
        valid_metrics['recall'] /= data_len
    return valid_loss, valid_metrics


def task_train(model, buffer, lifelong_datasets, config, metadata, logbook, dist_args=None):
    distributed = dist_args is not None
    if distributed:
        gpu = dist_args["gpu"]
        rank = dist_args["rank"]
    else:
        gpu = None
        rank = 0

    best_checkpoint = {
        "model_state_dict": deepcopy(model.method_state_dict()),
        "best_modified_jaccard": 0
    }
    best_checkpoint_file = os.path.join(config['logging_path'], "best_checkpoint")
    if config['use_best_model']:
        if config['task_epoch'] > 0 and os.path.exists(best_checkpoint_file):
            if distributed:
                best_checkpoint = torch.load(best_checkpoint_file, map_location=f"cuda:{dist_args['gpu']}")
            else:
                best_checkpoint = torch.load(best_checkpoint_file)

    task_train_data = lifelong_datasets['train']
    if config["method"] == "agem":
        bsm = 0.0
        task_train_data_with_buffer = TaskDataMergedWithBuffer(buffer, task_train_data, buffer_sampling_multiplier=bsm)
    else:
        bsm = config["buffer_sampling_multiplier"]
        task_train_data_with_buffer = TaskDataMergedWithBuffer(buffer, task_train_data, buffer_sampling_multiplier=bsm)
    task_valid_data = lifelong_datasets['intask_valid']
    cur_task_id = task_train_data.cur_task_id

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            task_train_data_with_buffer, num_replicas=dist_args["world_size"], rank=rank)
    else:
        train_sampler = None

    train_loader = data.DataLoader(
        task_train_data_with_buffer, batch_size=config["batch_size"], shuffle=(train_sampler is None),
        num_workers=config["num_workers"], pin_memory=True, sampler=train_sampler
    )
    valid_loader = data.DataLoader(
        task_valid_data, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"],
        pin_memory=True
    )

    if cur_task_id == 0:
        num_epochs = config['epochs_per_task'] * 2
        print_msg(f"Training for {num_epochs} epochs for the first task (double that the other tasks)")
    else:
        num_epochs = config['epochs_per_task']
    print_msg(f"Starting training of task {cur_task_id} epoch {config['task_epoch']} till epoch {num_epochs}")
    for epoch in range(config['task_epoch'], num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        start_time = time.time()
        log_dict = {}
        train_loss, train_metrics = epoch_train(model, train_loader, config, metadata, gpu, rank)
        log_dict[f"train_loss_{cur_task_id}"] = train_loss
        for metric in train_metrics.keys():
            log_dict[f"train_{metric}_{cur_task_id}"] = train_metrics[metric]

        valid_loss, valid_metrics = evaluate(model, valid_loader, config, metadata, test_mode=False, gpu=gpu)
        log_dict[f"valid_loss_{cur_task_id}"] = valid_loss
        for metric in valid_metrics.keys():
            log_dict[f"valid_{metric}_{cur_task_id}"] = valid_metrics[metric]

        model.net.eval()
        model.consolidate_epoch_knowledge(log_dict[f"valid_modified_jaccard_{cur_task_id}"],
                                          task_data=task_train_data,
                                          device=config["device"],
                                          batch_size=config["batch_size"])
        # If using the lmdb database, close it and open a new environment to kill active readers
        buffer.reset_lmdb_database()

        if config['use_best_model']:
            if log_dict[f"valid_modified_jaccard_{cur_task_id}"] >= best_checkpoint["best_modified_jaccard"]:
                best_checkpoint["best_modified_jaccard"] = log_dict[f"valid_modified_jaccard_{cur_task_id}"]
                best_checkpoint["model_state_dict"] = deepcopy(model.method_state_dict())
            log_dict[f"best_valid_modified_jaccard_{cur_task_id}"] = best_checkpoint["best_modified_jaccard"]

        if distributed:
            dist.barrier() #to calculate the time based on the slowest gpu
        end_time = time.time()
        log_dict[f"elapsed_time"] =  round(end_time - start_time, 2)

        if rank == 0:
            utils.log(epoch, cur_task_id, log_dict, logbook)
            if distributed:
                dist.barrier()
                log_dict["rank"] = rank
                print_msg(log_dict)
        else:
            dist.barrier()
            log_dict["rank"] = rank
            print_msg(log_dict)

        # Checkpointing
        config["task_epoch"] = epoch + 1
        if (config["task_epoch"] % config['checkpoint_interval']) == 0 and rank == 0:
            print_msg("Saving latest checkpoint")
            save_file = os.path.join(config['logging_path'], "latest_model")
            lifelong_methods.utils.save_model(save_file, config, metadata, model, buffer, lifelong_datasets)
            if config['use_best_model']:
                print_msg("Saving best checkpoint")
                torch.save(best_checkpoint, best_checkpoint_file)

    # reset the model parameters to the best performing model
    if config['use_best_model']:
        model.load_method_state_dict(best_checkpoint["model_state_dict"])


def tasks_eval(model, dataset, cur_task_id, config, metadata, logbook, dataset_type="valid", dist_args=None):
    """log the accuracies of the new model on all observed tasks
    :param metadata:
    """
    assert dataset.complete_information_mode is True

    distributed = dist_args is not None
    if distributed:
        gpu = dist_args["gpu"]
        rank = dist_args["rank"]
    else:
        gpu = None
        rank = 0

    metrics_dict = {}
    for task_id in range(cur_task_id + 1):
        dataset.choose_task(task_id)
        dataloader = data.DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
        )
        _, metrics = evaluate(model, dataloader, config, metadata, test_mode=True, gpu=gpu)
        for metric in metrics.keys():
            metrics_dict[f"task_{task_id}_{dataset_type}_{metric}"] = metrics[metric]
    dataset.load_tasks_up_to(cur_task_id)
    dataloader = data.DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True
    )
    _, metrics = evaluate(model, dataloader, config, metadata, test_mode=True, gpu=gpu)
    for metric in metrics.keys():
        metrics_dict[f"average_{dataset_type}_{metric}"] = metrics[metric]

    if rank == 0:
        utils.log_task(cur_task_id, metrics_dict, logbook)
        if distributed:
            dist.barrier()
            metrics_dict["rank"] = rank
            print_msg(metrics_dict)
    else:
        dist.barrier()
        metrics_dict["rank"] = rank
        print_msg(metrics_dict)
