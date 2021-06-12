Federated Adversarial Debiasing (FADE)
======================================

![FADE](assets/fade.png)

Clone the repository and setup the environment.
```shell
git clone git@github.com:illidanlab/FADE.git
cd FADE
# create conda env
conda env create -f conda.yml
conda activate fade
# run
python -m fade.mainx
```

To run repeated experiments, we use `wandb` to log. Run
```shell
wandb sweep <sweep.yaml>
```
Note, you need a wandb account which will be required at first run.

## Prepare

### Dataset

* **Office**: Download zip file from [here](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view) (preprocessed by [SHOT](https://github.com/tim-learn/SHOT)) and unpack into `./data/office31`. Verify the file structure to make sure the missing image path exist.
* **OfficeHome**: Download zip file from [here](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view) (preprocessed by [SHOT](https://github.com/tim-learn/SHOT)) and unpack into `./data/OfficeHome65`. Verify the file structure to make sure the missing image path exist.

### Pre-trained source-domain models

For each UDA tasks, we pre-train models on the source domain first. You can pre-train these models by yourself:
```shell
source sweeps/Office31_UDA/A_fedavg.sh
```
Instead, you may download the pre-trained source-domain models from [here](https://www.dropbox.com/sh/phuusbtbxd7r2fa/AAAGbFpmzn4wkAMF0utVCi28a?dl=0). Place under `out/models/`.  

### Pre-trained adapted models

To add soon.

## Run UDA experiments

* Office dataset
    ```shell
    # pretrain the model on domain A, D, W.
    source sweeps/Office31_UDA/A_fedavg.sh
    # create wandb sweeps for A2X, D2X, W2X where X is one of the rest two domains.
    # the command will prompt the agent commands.
    source sweeps/Office31_UDA/sweep_all.sh
    # Run wandb agent commands from the prompt or the sweep page.
    wandb agent <agent id>
    ```
    Demo wandb project page: [fade-demo-Office31_X2X_UDA](https://wandb.ai/jyhong/fade-demo-Office31_X2X_UDA?workspace=user-jyhong). Check [sweeps](https://wandb.ai/jyhong/fade-demo-Office31_X2X_UDA/sweeps?workspace=user-jyhong) here.
* OfficeHome dataset
    ```shell
    # pretrain the model on domain R
    source sweeps/OfficeHome65_1to3_uda_iid/R_fedavg.sh
    # create wandb sweeps for R2X where X is one of the rest domains.
    # the command will prompt the agent commands.
    source sweeps/OfficeHome65_1to3_uda_iid/sweep_all.sh
    # Run wandb agent commands from the prompt or the sweep page.
    wandb agent <agent id>
    ```

## Extend with other UDA methods
