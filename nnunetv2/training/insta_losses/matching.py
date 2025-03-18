import torch
from torch import nn

from nnunetv2.training.insta_losses.helpers import gpu_connected_components, get_regions
from nnunetv2.training.insta_losses.helpers import dice

class RegionDiceLoss(nn.Module):
    def __init__(self):
        super(RegionDiceLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, y):

        y = y.squeeze(1)
        multi_label = torch.zeros_like(y)
        for i in range(x.shape[0]):
            multi_label[i], _ = gpu_connected_components(y[i])
            
        multi_label = multi_label.unsqueeze(1)
        y = y.unsqueeze(1)

        x = x.to(y.device)
        multi_label = multi_label.to(y.device)

        # Ensure inputs are proper probabilities
        x = torch.sigmoid(x) if x.dim() == 5 else x

        batch_size = x.size(0)
        losses = []

        for b in range(batch_size):
            x_volume = x[b].squeeze()  # (D, H, W)
            y_volume = multi_label[b].squeeze()

            region_map, num_features = get_regions(y_volume)

            if num_features == 0:
                # Handle cases with no regions
                losses.append(torch.tensor(1.0, device=self.device))
                continue

            region_dice_scores = []
            for region_label in range(1, num_features + 1):
                region_mask = (region_map == region_label)
                x_region = x_volume[region_mask]
                y_region = y_volume[region_mask]

                dice_score = dice(x_region, y_region)
                region_dice_scores.append(dice_score)

            # Calculate mean Dice score for this volume
            mean_dice = torch.mean(torch.stack(region_dice_scores))
            losses.append(1 - mean_dice)  # Convert to loss

        # Return mean loss across batch
        loss = torch.mean(torch.stack(losses))
        return loss