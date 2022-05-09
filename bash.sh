#!/usr/bin/env bash

export PYTHONWARNINGS="ignore"
export GPU=0
python3 cifar100.py --config config/cifar100.cfg --gpu ${GPU} --net resnet18
# python3 cifar100.py --config config/cifar100_2.cfg --gpu ${GPU}
# python3 cifar100.py --config config/cifar100_3.cfg --gpu ${GPU}
# python3 cifar100.py --config config/cifar100_4.cfg --gpu ${GPU}
# python3 cifar100.py --config config/cifar100_5.cfg --gpu ${GPU}
# python3 cifar100.py --config config/cifar100_6.cfg --gpu ${GPU}
# python3 cifar100.py --config config/cifar100_7.cfg --gpu ${GPU}