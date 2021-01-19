import numpy as np
import torch.nn as nn
import json


def log(epoch, task_id, log_dict, logbook):
    log_dict["message"] = f"task_{task_id}_metrics"
    log_dict["task_id"] = task_id
    log_dict["task_epoch"] = epoch
    log_dict["step"] = epoch
    logbook.write_metric(log_dict)


def log_task(task_id, log_dict, logbook):
    log_dict["message"] = f"incremental_metrics"
    log_dict["task_id"] = task_id
    log_dict["step"] = task_id
    logbook.write_metric(log_dict)


def pad_random_crop(tensor_img, per_direction_padding=0):
    pad_left = pad_right = pad_top = pad_bottom = per_direction_padding
    tensor_width = tensor_img.shape[-1]
    tensor_height = tensor_img.shape[-2]
    tensor_img = nn.functional.pad(tensor_img,
                                   [pad_left, pad_right, pad_top, pad_bottom])

    start_index_width = np.random.randint(0, pad_left + pad_right)
    start_index_height = np.random.randint(0, pad_top + pad_bottom)
    end_index_width = start_index_width + tensor_width
    end_index_height = start_index_height + tensor_height

    return tensor_img[..., start_index_height:end_index_height, start_index_width:end_index_width]


def random_horizontal_flip(tensor_img, flip_prop=0.5):
    do_flip = np.random.random() >= (1 - flip_prop)
    if do_flip:
        return tensor_img.flip((-1))
    else:
        return tensor_img


def remove_extra_logs(cur_task_id, cur_epoch, file):
    logs_to_keep = []
    remove_task_summary = False
    with open(file, 'r') as logs_file:
        for line in logs_file:
            json_line = json.loads(line)
            if not (json_line['logbook_type'] == "metric"):
                logs_to_keep.append(json_line)
            elif json_line["task_id"] < cur_task_id:
                logs_to_keep.append(json_line)
            elif json_line["task_id"] == cur_task_id:
                if "task_epoch" in json_line.keys() and json_line["task_epoch"] < cur_epoch:
                    logs_to_keep.append(json_line)
                elif "task_epoch" in json_line.keys() and json_line["task_epoch"] >= cur_epoch:
                    remove_task_summary = True
                elif not remove_task_summary:
                    logs_to_keep.append(json_line)
    with open(file, 'w') as logs_file:
        for json_line in logs_to_keep:
            logs_file.write(json.dumps(json_line))
            logs_file.write("\n")


def extend_list(input_, output_length):
    if isinstance(input_, int):
        output = [input_ for _ in range(output_length)]
    elif hasattr(input_, '__iter__'):
        if len(input_) < output_length:
            output = input_
            output.extend([input_[-1] for _ in range(output_length - len(input_))])
        elif len(input_) > output_length:
            output = input_[:output_length]
        else:
            output = input_
    else:
        raise TypeError("Neither an integer nor an iterable was provided")
    return output