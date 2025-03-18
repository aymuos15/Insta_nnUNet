import torch 
import numpy as np

from scipy.optimize import linear_sum_assignment

import cc3d

from nnunetv2.training.insta_losses.helpers import dice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##################
#! Panoptic Dice #
##################
#? The TP, FP and FN values are also used for calculating the detection rate. 
def panoptic_dice(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):

    mask_pred_tensor = torch.tensor(mask_pred, dtype=torch.long).to(device)
    mask_ref_tensor = torch.tensor(mask_ref, dtype=torch.long).to(device)
    
    pred_cc = torch.zeros_like(mask_pred_tensor)
    gt_cc = torch.zeros_like(mask_ref_tensor)

    # running the connected components for pred
    for batch_idx in range(mask_pred.shape[0]):
        pred_batch = mask_pred[batch_idx].copy()  # Use numpy copy
        pred_cc_batch = cc3d.connected_components(pred_batch, connectivity=26)
        pred_cc_batch = pred_cc_batch.astype(np.int32)
        pred_cc[batch_idx] = torch.tensor(pred_cc_batch, dtype=torch.long).to(device)

    # running the connected components for gt
    for batch_idx in range(mask_ref.shape[0]):
        gt_batch = mask_ref[batch_idx].copy()  # Use numpy copy
        gt_cc_batch = cc3d.connected_components(gt_batch, connectivity=26)
        gt_cc_batch = gt_cc_batch.astype(np.int32)
        gt_cc[batch_idx] = torch.tensor(gt_cc_batch, dtype=torch.long).to(device)
    
    matches = create_match_dict(pred_cc, gt_cc)
    match_data = get_all_matches(matches)

    fp = sum(1 for pred, gt, _ in match_data if gt is None)
    fn = sum(1 for pred, gt, _ in match_data if pred is None)

    optimal_matches = optimal_matching(match_data)

    tp = len(optimal_matches)
    
    if tp == 0:
        return 0.0, 0

    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    sq = sum(score for _, _, score in optimal_matches) / tp

    return rq * sq, tp, fp, fn

def create_match_dict(pred_label_cc, gt_label_cc):
    pred_to_gt = {}
    gt_to_pred = {}
    dice_scores = {}

    pred_labels = torch.unique(pred_label_cc)[1:]  # Exclude background (0)
    gt_labels = torch.unique(gt_label_cc)[1:]  # Exclude background (0)

    pred_masks = {label.item(): pred_label_cc == label for label in pred_labels}
    gt_masks = {label.item(): gt_label_cc == label for label in gt_labels}

    for pred_item, pred_mask in pred_masks.items():
        for gt_item, gt_mask in gt_masks.items():
            if torch.any(torch.logical_and(pred_mask, gt_mask)):
                pred_to_gt.setdefault(pred_item, []).append(gt_item)
                gt_to_pred.setdefault(gt_item, []).append(pred_item)
                dice_scores[(pred_item, gt_item)] = dice(pred_mask, gt_mask)

    for gt_item in gt_labels:
        gt_to_pred.setdefault(gt_item.item(), [])
    for pred_item in pred_labels:
        pred_to_gt.setdefault(pred_item.item(), [])

    return {"pred_to_gt": pred_to_gt, "gt_to_pred": gt_to_pred, "dice_scores": dice_scores}

def get_all_matches(matches):
    match_data = []

    for gt, preds in matches["gt_to_pred"].items():
        if not preds:
            match_data.append((None, gt, 0.0))
        else:
            for pred in preds:
                dice_score = matches["dice_scores"].get((pred, gt), 0.0)
                match_data.append((pred, gt, dice_score))

    for pred, gts in matches["pred_to_gt"].items():
        if not gts:
            match_data.append((pred, None, 0.0))

    return match_data

def optimal_matching(match_data):
    predictions = set()
    ground_truths = set()
    valid_matches = []

    for pred, gt, score in match_data:
        if pred is not None and gt is not None:
            predictions.add(pred)
            ground_truths.add(gt)
            valid_matches.append((pred, gt, score))

    pred_to_index = {pred: i for i, pred in enumerate(predictions)}
    gt_to_index = {gt: i for i, gt in enumerate(ground_truths)}

    cost_matrix = torch.ones((len(predictions), len(ground_truths)))

    for pred, gt, score in valid_matches:
        i, j = pred_to_index[pred], gt_to_index[gt]
        cost_matrix[i, j] = 1 - score

    #todo: Use a torch variant here?
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())

    optimal_matches = []
    for i, j in zip(row_ind, col_ind):
        pred = list(predictions)[i]
        gt = list(ground_truths)[j]
        score = 1 - cost_matrix[i, j].item()
        optimal_matches.append((pred, gt, score))

    return optimal_matches