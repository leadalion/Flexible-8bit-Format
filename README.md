# Exploring the Potential of Flexible 8-bit Format: Design and Algorithm

## Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

## MMlab Network Config Prepare

For convenience, we have introduced the mmlab framework into the code. If you have problems with the installation of the openmmlab framework, you can refer to the link https://mmcv.readthedocs.io/en/latest/get_started/installation.html

And before you use it, you should prepare dataset like ImageNet, COCO2017, and CityScapes. Organized as below

```bash
dataset_root
└── imagenet
    ├── meta
    ├── train
    └── val 
```
Take resnet50 as an example:

```bash
cd ${code_root}
# dataset_root should end with '/'
sh scripts/mmlab/download_network_config/cls/res50.sh ${dataset_root}
```

## Bert Config Prepare
Download bert-base-uncased models from https://huggingface.co/bert-base-uncased

## Running Quantization Experiments

```bash
# mmlab model
sh script/mix_all_8bit_script/cls/res50.sh
```

```bash
# bert model
sh script/glue/run.sh
```