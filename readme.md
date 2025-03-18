# Changes

## 1. Losses
A. Add them under training
`/home/localssk23/nnUNet/nnunetv2/training/nnUNetTrainer/variants/insta_losses/nnUNetTrainerRegionDice_CELoss.py`

B. Then to call them, add them under variants of the trainer class.
    `/home/localssk23/nnUNet/nnunetv2/training/nnUNetTrainer/variants/insta_losses/nnUNetTrainerRegionDice_CELoss.py`

## 2. Metrics

## 3. Logging

Extra:
GradScaler: `https://github.com/MIC-DKFZ/nnUNet/issues/2742`