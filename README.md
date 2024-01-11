# Class-Aware Pruning for Efficient Neural Networks
This repository includes our code for the paper 'Class-Aware Pruning for Efficient Neural Networks' in DATE 2024.
It is a novel structured network pruning method aimed at enhancing the efficiency of DNNs. 
The code is currently being progressively open-sourced.


## Introduction
In this paper, we introduce a class-aware pruning technique for deep neural networks (DNNs) that addresses the high computational cost associated with their large number of floating-point operations (FLOPs), a significant challenge in resource-constrained environments like edge devices. Unlike traditional pruning methods focused on weights, gradients, and activations, our approach modifies the training process to assess and remove filters important for only a limited number of classes. This iterative pruning and retraining process continues until no further filters can be removed, ensuring the retention of filters crucial for multiple classes. Our technique surpasses existing methods in accuracy, pruning ratio, and FLOPs reduction, effectively decreasing the number of weights and FLOPs while maintaining high inference accuracy.

## Environment Setup
This code is tested with Python 3.9, Pytorch = 2.0.1 and CUDA = 11.7

```
pip install -r requirements.txt
```
