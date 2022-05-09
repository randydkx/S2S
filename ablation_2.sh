#!/usr/bin/env bash

export PYTHONWARNINGS="ignore"
export GPU=1

# python cifar100.py --config config/cifar100.cfg --gpu ${GPU} --net wideresnet
python cifar100.py --config config/cifar100.cfg --gpu ${GPU} --net resnet18
python cifar100.py --config config/cifar100.cfg --gpu ${GPU} --net preactresnet