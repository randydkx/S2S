#!/usr/bin/env bash

export PYTHONWARNINGS="ignore"
export GPU=0

python cifar100.py --config config/cifar100.cfg --gpu ${GPU} --lambda_e 2.0
python cifar100.py --config config/cifar100.cfg --gpu ${GPU} --lambda_con 0.5
python cifar100.py --config config/cifar100.cfg --gpu ${GPU} --lambda_con 2.0
python cifar100.py --config config/cifar100.cfg --gpu ${GPU} --lambda_id 0.5
python cifar100.py --config config/cifar100.cfg --gpu ${GPU} --lambda_id 2.0
