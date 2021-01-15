import torch
import numpy as np


def jaccard_sim(predictions, target):
    '''
    Jaccard similarity (intersection over union)
    Args:
        predictions (boolean tensor): Model predictions of size batch_size x number of classes
        target (boolean tensor): target predictions of size batch_size x number of classes
    '''
    jac_sim = torch.sum(target & predictions, dim=-1) * 1.0 / torch.sum(target | predictions, dim=-1)
    jac_sim = torch.sum(jac_sim)
    jac_sim /= target.shape[0]
    return jac_sim.item()


def modified_jaccard_sim(predictions, target):
    '''
    Jaccard similarity multiplied by percentage of correct predictions (per sample precision)
    Args:
        predictions (boolean tensor): Model predictions of size batch_size x number of classes
        target (boolean tensor): target predictions of size batch_size x number of classes
    '''
    jac_sim = torch.sum(target & predictions, dim=-1) * 1.0 / torch.sum(target | predictions, dim=-1)
    correct_pred_pct = torch.sum(target & predictions, dim=-1) * 1.0 / (torch.sum(predictions, dim=-1) + 1e-8)
    modified_jac_sim = jac_sim * correct_pred_pct
    modified_jac_sim = torch.sum(modified_jac_sim)
    modified_jac_sim /= target.shape[0]
    return modified_jac_sim.item()


def strict_accuracy(predictions, target):
    '''
    The accuracy measure where if some of the labels for a sample are predicted correctly, and some are wrong, it gives
    a score of zero to the accuracy of that sample
    Args:
        predictions (boolean tensor): Model predictions of size batch_size x number of classes
        target (boolean tensor): target predictions of size batch_size x number of classes
    '''
    acc = torch.sum((target == predictions).all(dim=-1)) * 1.0
    acc /= target.shape[0]
    return acc.item()


def recall(predictions, target):
    '''
    The recall measure
    Args:
        predictions (torch.BoolTensor): Model predictions of size batch_size x number of classes
        target (torch.BoolTensor): target predictions of size batch_size x number of classes
    '''
    recall = torch.sum(predictions[target]) * 1.0 / torch.sum(target)
    return recall.item()


