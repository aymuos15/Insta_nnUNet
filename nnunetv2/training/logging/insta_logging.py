import torch 
import numpy as np

import cc3d

from nnunetv2.evaluation.insta_metrics import create_match_dict, get_all_matches, optimal_matching

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def panoptic_scores(net_output, gt):
    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device)
            y_onehot.scatter_(1, gt.long(), 1)
    
    labelled_pred = torch.zeros_like(net_output)
    labelled_gt = torch.zeros_like(y_onehot)

    for batch in range(net_output.shape[0]):
        for channel in range(net_output.shape[1]):
            components = cc3d.connected_components(net_output[batch, channel].cpu().numpy(), connectivity=26)
            components = components.astype(np.uint8)
            labelled_pred[batch, channel] = torch.tensor(components, device=net_output.device)
    
    for batch in range(y_onehot.shape[0]):
        for channel in range(y_onehot.shape[1]):
            components = cc3d.connected_components(y_onehot[batch, channel].cpu().numpy(), connectivity=26)
            components = components.astype(np.uint8)
            labelled_gt[batch, channel] = torch.tensor(components, device=net_output.device)

    pred_label_cc = net_output
    gt_label_cc = y_onehot

    matches = create_match_dict(pred_label_cc, gt_label_cc)
    match_data = get_all_matches(matches)

    fp = sum(1 for pred, gt, _ in match_data if gt is None)
    fn = sum(1 for pred, gt, _ in match_data if pred is None)

    optimal_matches = optimal_matching(match_data)

    tp = len(optimal_matches)
    
    if tp == 0:
        return 0.0

    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    sq = sum(score for _, _, score in optimal_matches) / tp

    return rq * sq, tp, fp, fn