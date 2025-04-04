import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

import cupy as cp
from cucim.skimage import measure as cucim_measure

from scipy.ndimage import distance_transform_edt

#############################
#! GPU connected components #
#############################
#? https://github.com/aymuos15/GPU-Connected-Components
def gpu_connected_components(img, connectivity=None):

    img_cupy = cp.asarray(img)
    labeled_img, num_features = cucim_measure.label(img_cupy, connectivity=connectivity, return_num=True)
    labeled_img_torch = torch.as_tensor(labeled_img, device=img.device)

    return labeled_img_torch, num_features

###################################
#! Distance transform calculation #
###################################
def get_regions(self, gt):
    # Step 1: Connected Components for 3D volume
    labeled_array, num_features = self.get_connected_components(gt.cpu())

    # Step 2: Compute distance transform for each 3D region
    distance_map = torch.zeros_like(gt, dtype=torch.float32)
    region_map = torch.zeros_like(gt, dtype=torch.long)

    for region_label in range(1, num_features + 1):
        region_mask = (labeled_array == region_label)

        # Convert to numpy for distance transform
        region_mask_np = region_mask.cpu().numpy()
        distance = torch.from_numpy(
            distance_transform_edt(~region_mask_np)
        ).to(self.device)

        if region_label == 1 or distance_map.max() == 0:
            distance_map = distance
            region_map = region_label * torch.ones_like(gt, dtype=torch.long)
        else:
            update_mask = distance < distance_map
            distance_map[update_mask] = distance[update_mask]
            region_map[update_mask] = region_label

    return region_map, num_features

#########
#! Dice #
#########
def dice(im1, im2):
    
    # Ensure the tensors are on the same device
    im1 = im1.to(im2.device)
    im2 = im2.to(im1.device)

    # Compute Dice coefficient using optimized operations
    intersection = torch.sum(im1 * im2)
    im1_sum = torch.sum(im1)
    im2_sum = torch.sum(im2)
    
    dice_score = (2. * intersection) / (im1_sum + im2_sum)

    return dice_score

############
#! Tversky #
############
def tversky(im1, im2, fp=0.3, fn=0.7):
    
    # Ensure the tensors are on the same device
    im1 = im1.to(im2.device)
    im2 = im2.to(im1.device)

    # Compute Tversky index using specified fp and fn weights
    intersection = torch.sum(im1 * im2)
    false_positive = torch.sum(im1 * (1 - im2))
    false_negative = torch.sum((1 - im1) * im2)
    
    tversky_score = intersection / (intersection + fp * false_positive + fn * false_negative)

    return tversky_score

##############
#! Blob Loss #
##############
def vprint(*args):
    verbose = False
    if verbose:
        print(*args)


def compute_compound_loss(
    criterion_dict: dict,
    raw_network_outputs: torch.Tensor,
    label: torch.Tensor,
    blob_loss_mode=False,
    masked=True,
):
    """
    This computes a compound loss by looping through a criterion dict!
    """
    # vprint("outputs:", outputs)
    losses = []
    for entry in criterion_dict.values():
        name = entry["name"]
        vprint("loss name:", name)
        criterion = entry["loss"]
        weight = entry["weight"]

        sigmoid = entry["sigmoid"]
        if blob_loss_mode == False:
            vprint("computing main loss!")
            if sigmoid == True:
                sigmoid_network_outputs = torch.sigmoid(raw_network_outputs)
                individual_loss = criterion(sigmoid_network_outputs, label)
            else:
                individual_loss = criterion(raw_network_outputs, label)
        elif blob_loss_mode == True:
            vprint("computing blob loss!")
            if masked == True:  # this is the default blob loss
                if sigmoid == True:
                    sigmoid_network_outputs = torch.sigmoid(raw_network_outputs)
                    individual_loss = compute_blob_loss_multi(
                        criterion=criterion,
                        network_outputs=sigmoid_network_outputs,
                        multi_label=label,
                    )
                else:
                    individual_loss = compute_blob_loss_multi(
                        criterion=criterion,
                        network_outputs=raw_network_outputs,
                        multi_label=label,
                    )
            elif masked == False:  # without masking for ablation study
                if sigmoid == True:
                    sigmoid_network_outputs = torch.sigmoid(raw_network_outputs)
                    individual_loss = compute_no_masking_multi(
                        criterion=criterion,
                        network_outputs=sigmoid_network_outputs,
                        multi_label=label,
                    )
                else:
                    individual_loss = compute_no_masking_multi(
                        criterion=criterion,
                        network_outputs=raw_network_outputs,
                        multi_label=label,
                    )

        weighted_loss = individual_loss * weight
        losses.append(weighted_loss)

    vprint("losses:", losses)
    loss = sum(losses)
    return loss


def compute_blob_loss_multi(
    criterion,
    network_outputs: torch.Tensor,
    multi_label: torch.Tensor,
):
    """
    1. loop through elements in our batch
    2. loop through blobs per element compute loss and divide by blobs to have element loss
    2.1 we need to account for sigmoid and non/sigmoid in conjunction with BCE
    3. divide by batch length to have a correct batch loss for back prop
    """
    batch_length = multi_label.shape[0]
    vprint("batch_length:", batch_length)

    element_blob_loss = []
    # loop over elements
    for element in range(batch_length):
        if element < batch_length:
            end_index = element + 1
        elif element == batch_length:
            end_index = None

        element_label = multi_label[element:end_index, ...]
        vprint("element label shape:", element_label.shape)

        vprint("element_label:", element_label.shape)

        element_output = network_outputs[element:end_index, ...]

        # loop through labels
        unique_labels = torch.unique(element_label)
        blob_count = len(unique_labels) - 1
        vprint("found this amount of blobs in batch element:", blob_count)

        label_loss = []
        for ula in unique_labels:
            if ula == 0:
                vprint("ula is 0 we do nothing")
            else:
                # first we need one hot labels
                vprint("ula greater than 0:", ula)
                label_mask = element_label > 0
                # we flip labels
                label_mask = ~label_mask

                # we set the mask to true where our label of interest is located
                # vprint(torch.count_nonzero(label_mask))
                label_mask[element_label == ula] = 1
                # vprint(torch.count_nonzero(label_mask))
                vprint("label_mask", label_mask)
                # vprint("torch.unique(label_mask):", torch.unique(label_mask))

                the_label = element_label == ula
                the_label_int = the_label.int()
                vprint("the_label:", torch.count_nonzero(the_label))


                # debugging
                # masked_label = the_label * label_mask
                # vprint("masked_label:", torch.count_nonzero(masked_label))

                masked_output = element_output * label_mask

                try:
                    # we try with int labels first, but some losses require floats
                    blob_loss = criterion(masked_output, the_label_int)
                except:
                    # if int does not work we try float
                    blob_loss = criterion(masked_output, the_label.float())
                vprint("blob_loss:", blob_loss)

                label_loss.append(blob_loss)

        # compute mean
        vprint("label_loss:", label_loss)
        # mean_label_loss = 0
        vprint("blobs in crop:", len(label_loss))
        if not len(label_loss) == 0:
            mean_label_loss = sum(label_loss) / len(label_loss)
            # mean_label_loss = sum(label_loss) / \
            #     torch.count_nonzero(label_loss)
            vprint("mean_label_loss", mean_label_loss)
            element_blob_loss.append(mean_label_loss)

    # compute mean
    vprint("element_blob_loss:", element_blob_loss)
    mean_element_blob_loss = 0
    vprint("elements in batch:", len(element_blob_loss))
    if not len(element_blob_loss) == 0:
        mean_element_blob_loss = sum(element_blob_loss) / len(element_blob_loss)
        # element_blob_loss) / torch.count_nonzero(element_blob_loss)

    vprint("mean_element_blob_loss", mean_element_blob_loss)

    return mean_element_blob_loss


def compute_no_masking_multi(
    criterion,
    network_outputs: torch.Tensor,
    multi_label: torch.Tensor,
):
    """
    1. loop through elements in our batch
    2. loop through blobs per element compute loss and divide by blobs to have element loss
    2.1 we need to account for sigmoid and non/sigmoid in conjunction with BCE
    3. divide by batch length to have a correct batch loss for back prop
    """
    batch_length = multi_label.shape[0]
    vprint("batch_length:", batch_length)

    element_blob_loss = []
    # loop over elements
    for element in range(batch_length):
        if element < batch_length:
            end_index = element + 1
        elif element == batch_length:
            end_index = None

        element_label = multi_label[element:end_index, ...]
        vprint("element label shape:", element_label.shape)

        vprint("element_label:", element_label.shape)

        element_output = network_outputs[element:end_index, ...]

        # loop through labels
        unique_labels = torch.unique(element_label)
        blob_count = len(unique_labels) - 1
        vprint("found this amount of blobs in batch element:", blob_count)

        label_loss = []
        for ula in unique_labels:
            if ula == 0:
                vprint("ula is 0 we do nothing")
            else:
                # first we need one hot labels
                vprint("ula greater than 0:", ula)

                the_label = element_label == ula
                the_label_int = the_label.int()

                vprint("the_label:", torch.count_nonzero(the_label))

                # we compute the loss with no mask
                try:
                    # we try with int labels first, but some losses require floats
                    blob_loss = criterion(element_output, the_label_int)
                except:
                    # if int does not work we try float
                    blob_loss = criterion(element_output, the_label.float())
                vprint("blob_loss:", blob_loss)

                label_loss.append(blob_loss)

            # compute mean
            vprint("label_loss:", label_loss)
            # mean_label_loss = 0
            vprint("blobs in crop:", len(label_loss))
            if not len(label_loss) == 0:
                mean_label_loss = sum(label_loss) / len(label_loss)
                # mean_label_loss = sum(label_loss) / \
                #     torch.count_nonzero(label_loss)
                vprint("mean_label_loss", mean_label_loss)
                element_blob_loss.append(mean_label_loss)

    # compute mean
    vprint("element_blob_loss:", element_blob_loss)
    mean_element_blob_loss = 0
    vprint("elements in batch:", len(element_blob_loss))
    if not len(element_blob_loss) == 0:
        mean_element_blob_loss = sum(element_blob_loss) / len(element_blob_loss)
        # element_blob_loss) / torch.count_nonzero(element_blob_loss)

    vprint("mean_element_blob_loss", mean_element_blob_loss)

    return mean_element_blob_loss


def compute_loss(
    blob_loss_dict: dict,
    criterion_dict: dict,
    blob_criterion_dict: dict,
    raw_network_outputs: torch.Tensor,
    binary_label: torch.Tensor,
    multi_label: torch.Tensor,
):
    """
    This function computes the total loss. It has a global main loss and the blob loss term which is computed separately for each connected component. The binary_label is the binarized label for the global part. The multi label features separate integer labels for each connected component.

    Example inputs should look like:

    blob_loss_dict = {
        "main_weight": 1,
        "blob_weight": 0,
    }

    criterion_dict = {
        "bce": {
            "name": "bce",
            "loss": BCEWithLogitsLoss(reduction="mean"),
            "weight": 1.0,
            "sigmoid": False,
        },
        "dice": {
            "name": "dice",
            "loss": DiceLoss(
                include_background=True,
                to_onehot_y=False,
                sigmoid=True,
                softmax=False,
                squared_pred=False,
            ),
            "weight": 1.0,
            "sigmoid": False,
        },
    }

    blob_criterion_dict = {
        "bce": {
            "name": "bce",
            "loss": BCEWithLogitsLoss(reduction="mean"),
            "weight": 1.0,
            "sigmoid": False,
        },
        "dice": {
            "name": "dice",
            "loss": DiceLoss(
                include_background=True,
                to_onehot_y=False,
                sigmoid=True,
                softmax=False,
                squared_pred=False,
            ),
            "weight": 1.0,
            "sigmoid": False,
        },
    }
    """

    main_weight = blob_loss_dict["main_weight"]
    blob_weight = blob_loss_dict["blob_weight"]

    # main loss
    if main_weight > 0:
        vprint("main_weight greater than zero:", main_weight)
        # vprint("main_label:", main_label)
        main_loss = compute_compound_loss(
            criterion_dict=criterion_dict,
            raw_network_outputs=raw_network_outputs,
            label=binary_label,
            blob_loss_mode=False,
        )

    if blob_weight > 0:
        vprint("blob_weight greater than zero:", blob_weight)
        blob_loss = compute_compound_loss(
            criterion_dict=blob_criterion_dict,
            raw_network_outputs=raw_network_outputs,
            label=multi_label,
            blob_loss_mode=True,
        )

    # final loss
    if blob_weight == 0 and main_weight > 0:
        vprint(
            "main_weight:",
            main_weight,
            "// blob_weight:",
            blob_weight,
            "// computing main loss only",
        )
        loss = main_loss
        blob_loss = 0

    elif main_weight == 0 and blob_weight > 0:
        vprint(
            "main_weight:",
            main_weight,
            "// blob_weight:",
            blob_weight,
            "// computing blob loss only",
        )
        loss = blob_loss
        main_loss = 0  # we set this to 0

    elif main_weight > 0 and blob_weight > 0:
        vprint(
            "main_weight:",
            main_weight,
            "// blob_weight:",
            blob_weight,
            "// computing blob loss",
        )
        loss = main_loss * main_weight + blob_loss * blob_weight

    else:
        vprint("defaulting to equal weighted blob loss")
        loss = main_loss + blob_loss

    vprint("blob loss:", blob_loss)
    vprint("main loss:", main_loss)
    vprint("effective loss:", loss)

    return loss, main_loss, blob_loss

###############
#! RegionLoss #
###############
#? This may be wrong: https://github.com/by-liu/SegLossBias
#? Check the issues.
class LossMode:
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"

EPS = 1e-10

def expand_onehot_labels(
    labels: torch.Tensor, 
    target_shape: torch.Size, 
    ignore_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        elif labels.dim() == 4:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2], inds[3]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    return bin_labels, valid_mask

def get_region_proportion(
    x: torch.Tensor, 
    valid_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if valid_mask is not None:
        if valid_mask.dim() == 4:
            x = torch.einsum("bcwh,bcwh->bcwh", x, valid_mask)
            cardinality = torch.einsum("bcwh->bc", valid_mask)
        else:
            x = torch.einsum("bcwh,bwh->bcwh", x, valid_mask)
            cardinality = torch.einsum("bwh->b", valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
    else:
        cardinality = x.shape[2] * x.shape[3]

    return (torch.einsum("bcwh->bc", x) + EPS) / (cardinality + EPS)

def get_region_proportion_3d(
    x: torch.Tensor, 
    valid_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if valid_mask is not None:
        if valid_mask.dim() == 5:
            x = torch.einsum("bcxyz,bcxyz->bcxyz", x, valid_mask)
            cardinality = torch.einsum("bcxyz->bc", valid_mask)
        else:
            x = torch.einsum("bcxyz,bxyz->bcxyz", x, valid_mask)
            cardinality = torch.einsum("bxyz->b", valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
    else:
        cardinality = x.shape[2] * x.shape[3] * x.shape[4]

    return (torch.einsum("bcxyz->bc", x) + EPS) / (cardinality + EPS)

class CompoundLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.,
        factor: float = 1.,
        step_size: int = 0,
        max_alpha: float = 100.,
        temp: float = 1.,
        ignore_index: int = 255,
        background_index: int = -1,
        weight: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        # valid_modes = {LossMode.BINARY, LossMode.MULTILABEL, LossMode.MULTICLASS}
        self.mode = LossMode.BINARY        
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.factor = factor
        self.step_size = step_size
        self.temp = temp
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.weight = weight

    def cross_entropy(
        self, 
        inputs: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        if self.mode == LossMode.MULTICLASS:
            return F.cross_entropy(
                inputs, labels, weight=self.weight, ignore_index=self.ignore_index
            )
        
        if labels.dim() == 3:
            labels = labels.unsqueeze(dim=1)
        return F.binary_cross_entropy_with_logits(inputs, labels.type(torch.float32))

    def adjust_alpha(self, epoch: int) -> None:
        if self.step_size == 0:
            return
        if (epoch + 1) % self.step_size == 0:
            self.alpha = min(self.alpha * self.factor, self.max_alpha)

    def get_gt_proportion(
        self,
        mode: str,
        labels: torch.Tensor,
        target_shape: torch.Size,
        ignore_index: int = 255
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode == LossMode.MULTICLASS:
            bin_labels, valid_mask = expand_onehot_labels(labels, target_shape, ignore_index)
        else:
            valid_mask = (labels >= 0) & (labels != ignore_index)
            bin_labels = labels.unsqueeze(dim=1) if labels.dim() == 3 else labels

        get_prop_fn = get_region_proportion_3d if bin_labels.dim() == 5 else get_region_proportion
        gt_proportion = get_prop_fn(bin_labels, valid_mask)
        return gt_proportion, valid_mask

    def get_pred_proportion(
        self,
        mode: str,
        logits: torch.Tensor,
        temp: float = 1.0,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mode == LossMode.MULTICLASS:
            preds = F.log_softmax(temp * logits, dim=1).exp()
        else:
            preds = F.logsigmoid(temp * logits).exp()

        get_prop_fn = get_region_proportion_3d if preds.dim() == 5 else get_region_proportion
        return get_prop_fn(preds, valid_mask)