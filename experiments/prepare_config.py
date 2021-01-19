import os
import random

def prepare_config(args):
    config = vars(args)
    config["cur_task_id"] = 0
    config["task_epoch"] = 0
    if args.wandb_project is not None and args.wandb_project != "None":
        config["wandb_log"] = True
    else:
        config["wandb_log"] = False

    if config["reduce_lr_on_plateau"]:
        config["lr_schedule"] = None

    if config['run_id'] is None:
        config['run_id'] = random.randint(0, 100000)

    if config["incremental_joint"]:
        assert config["complete_info"], "Use incremental_joint only with the complete_info option. Otherwise, " \
                                        "increase the buffer size"
        config["total_n_memories"] = 0
        config["n_memories_per_class"] = -1

    if config["joint"]:
        assert config["complete_info"], "Use joint only with the complete_info option."
        config["total_n_memories"] = 0
        config["n_memories_per_class"] = -1

    config['id'] = f"{config['dataset']}" \
                   f"_{config['method']}" \
                   f"_{config['optimizer']}" \
                   f"_lr{config['lr']}" \
                   f"_wd{config['weight_decay']}"
    if not config['joint']:
        config['id'] += f"_tci{config['tasks_configuration_id']}"
    if config["complete_info"]:
        config['id'] += "_complete_info"
    if config["incremental_joint"]:
        config['id'] += "_incremental_joint"
    elif config['joint']:
        config['id'] += "_joint"
    elif config['n_memories_per_class'] > -1:
        config['total_n_memories'] = -1
        if config['n_memories_per_class'] > 0:
            config['id'] += f"_nmpc{config['n_memories_per_class']}"
    else:
        config['n_memories_per_class'] = -1
        if config['total_n_memories'] > 0:
            config['id'] += f"_tnm{config['total_n_memories']}"
    if config['reduce_lr_on_plateau']:
        config['id'] += "_lr_on_plat"
    elif config["lr_schedule"] is not None:
        config['id'] += "_lr_sched"
    if config['use_best_model']:
        config['id'] += "_use_best"
    config['id'] += f"_nl{config['n_layers']}" \
                    f"_ept{config['epochs_per_task']}" \
                    f"_s{config['seed']}" \
                    f"_{config['run_id']}"

    config['logging_path'] = os.path.join(config['logging_path_root'], config['id'])
    return config