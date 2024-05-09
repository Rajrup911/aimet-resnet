# AIMET-Resnet
[AIMET](https://quic.github.io/aimet-pages/index.html), AI Model Efficiency Toolkit is a library that provides advanced model quantization and compression techniques for trained neural network models. It provides features that have been proven to improve the run-time performance of deep learning neural network models with lower compute and memory requirements and minimal impact on task accuracy.

# Table of Contents
- [Task](#Task)
- [Model Source](#ModelSource)
- [Model Description](#ModelDescription)
- [Dataset Used](#DatasetUsed)
- [Results](#Results)


## Task
1. Learn about AIMET methods and API used for PTQ 
2. Apply PTQ on ResNet Model (W8A8 & W8A16)

## Model Source
PyTorch Image Models (timm) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that aim to pull together a wide variety of SOTA models with ability to reproduce ImageNet training results.
- Model is fetched using `timm.create_model('model_name', pretrained=True, num_classes=num_classes)` command

## Model Description
1. Model Name: ResNet50
2. Model Variant in Timm Library: `resnet50.tv_in1k`
3. Input Shape: (1, 3, 224, 224)

## Dataset Used
1. Pre-Trained on ImageNet
2. Used 1000 samples for validation and 100 samples for Calibration

## Results
- PTQ Methods applied: [CLE, BN-Fold, Adaround]

| Type          | Top 1 Accuracy (FP32)  | Top 1 Accuracy (PTQ) | Top 5 Accuracy (FP32)  | Top 5 Accuracy (PTQ) |
| ------------- |:-------------:|:-----:|:-------------:|:-----:|
| resnet50_W8A8    | 89.50% | 89.40% | 97.40% | 97.20% | 
| resnet50_W8A16   | 89.50% | 89.50% | 97.40% | 97.20% |

## License



### Command to run full pipeline
```zsh
bash run.sh
```
