# Changes

## 1. Losses
A. Add the code for the losses themselves here:
`/home/localssk23/nnUNet/nnunetv2/training/insta_losses`

Ex: `/home/localssk23/nnUNet/nnunetv2/training/insta_losses/compound.py`

B. To call them, add them here:
`/home/localssk23/nnUNet/nnunetv2/training/nnUNetTrainer/variants/`

Ex: `/home/localssk23/nnUNet/nnunetv2/training/nnUNetTrainer/variants/insta_losses/nnUNetTrainerRegionDice_CELoss.py`

Sample command to call the above loss: `nnUNet_compile=F nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainerRegionDice_CELoss`

## 2. Metrics
A. Create a file to define your metrics here: 
`/home/localssk23/nnUNet/nnunetv2/evaluation`

B. To integrate them into the logging, add them within: 
`/home/localssk23/nnUNet/nnunetv2/evaluation/evaluate_predictions.py`

Note: Just doing this will not generate any extra output, that will be done below in the logging. However, do run once to check if everything is okay.


## 3. Logging
To log the final pred/val score:
Check comment in line 1345 in `/home/localssk23/nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`

Need to use the `print_to_log_file` function.

Extra:
GradScaler: `https://github.com/MIC-DKFZ/nnUNet/issues/2742`