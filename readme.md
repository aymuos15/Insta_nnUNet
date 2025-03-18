# Simple guidlines for simple changes within nnUNet

# 1. Losses

### A. Add loss code
* Logic here: `nnUNet/nnunetv2/training/insta_losses`
* Example file: `nnUNet/nnunetv2/training/insta_losses/compound.py`

### B. Implement loss variants
* call in some file here: `nnUNet/nnunetv2/training/nnUNetTrainer/variants/`
* Example: `nnUNet/nnunetv2/training/nnUNetTrainer/variants/insta_losses/nnUNetTrainerRegionDice_CELoss.py`

### Usage
* Sample command:
    ```
    nnUNet_compile=F nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainerRegionDice_CELoss
    ```

# 2. Metrics

* A. Create a file to define your metrics here: 
    * `nnUNet/nnunetv2/evaluation`

* B. To integrate them into the logging, add them within: 
    * `nnUNet/nnunetv2/evaluation/evaluate_predictions.py`

**<ins>Note:</ins>** Just doing this will not generate any extra output, that will be done below in the logging. However, do run once to check if everything is okay.


# 3. Logging

### A. Log final prediction/validation scores:
* Check in `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`
    * Specifically: `print_to_log_file` function

### B. Final Plots:
* Check in `nnUNet/nnunetv2/training/logging/nnunet_logger.py`
    * Specifically: `plot_progress_png` function

### C. Every Epoch Train logs
Currently done a bit weirdly cos the import is from a loss folder. I found it a bit easier to understand by the below:

* I. 
    * Create a file to define your metrics here: `/home/localssk23/nnUNet/nnunetv2/training/logging`
    * Example file: `/home/localssk23/nnUNet/nnunetv2/training/logging/insta_logging.py`

* II. 
    * Make multiple changes in: `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py` \
    --> Follow #!%%1, #!%%2 etc. comment style in the file. There are 6 steps.


# 4 Checkpointing

## Extra:
**<ins>GradScaler**</ins>: `https://github.com/MIC-DKFZ/nnUNet/issues/2742`

**<ins>debug.json**</ins>: Even if I use a custom loss, why does the debug show Dice + CE?

**<ins>mask consideration in logging metrics**</ins>: I do not do that now, check the args difference between `get_tp_fp_fn_tn` and `panoptic_scores`