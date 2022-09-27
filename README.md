# Federated Learning from Pre-Trained Models: A Contrastive Learning Approach

Implementation of the paper accepted by NeurIPS 2022: [Federated Learning from Pre-Trained Models: A Contrastive Learning Approach](https://arxiv.org/abs/2209.10083).

## Requirments
This code requires the following:
* Python >= 3.9
* PyTorch >= 1.10.2
* Torchvision 0.8.2
* Numpy 1.21.5
* tensorboardX

## Data Preparation
* Download the train and test datasets manually from the given links and put them under ```./data/``` directory.
* Experiments are run on [Digit-5](https://drive.google.com/file/d/1moBE_ASD5vIOaU8ZHm_Nsj0KAfX5T0Sf/view), [Office-10](https://drive.google.com/file/d/1gxhV5xRXQgC9AL4XexduH7hdxDng7bJ3/view), and [Domainnet](https://drive.google.com/file/d/1_dx2-YDdvnNlQ13DTgDnLoGvMZvMyccR/view) with [source data](http://csr.bu.edu/ftp/visda/2019/multi-source/).

## Pre-Trained Models Preparation
* Download the directory ```weight``` containing pre-trained foundation models from [HERE](https://drive.google.com/drive/folders/12fwBTyW881Q3n5tkhsv8qf2YweLh-wWu) and put it under ```./lib/model``` directory.

## Running examples
* To train on Digit-5 with one backbone under the feature shift non-IID setting:
```
python exps/federated_main.py --alg fedavg    --dataset digit --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 >digit_fedavg_fnli_1bb_5u.log
python exps/federated_main.py --alg local     --dataset digit --num_users 5 --rounds 100 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 >digit_local_fnli_1bb_5u.log
python exps/federated_main.py --alg fedpcl    --dataset digit --num_users 5 --rounds 100 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 >digit_fedpcl_fnli_1bb_5u.log
```
* To train on Digit-5 with three backbones under the feature shift non-IID setting:
```
python exps/federated_main.py --alg fedavg    --dataset digit --num_users 5 --rounds 200 --num_bb 3 --feature_iid 1 --label_iid 0 --alpha 1 >digit_fedavg_filn_3bb_5u_a1.log
python exps/federated_main.py --alg local     --dataset digit --num_users 5 --rounds 100 --num_bb 3 --feature_iid 1 --label_iid 0 --alpha 1 >digit_local_filn_3bb_5u_a1.log
python exps/federated_main.py --alg fedpcl    --dataset digit --num_users 5 --rounds 50  --num_bb 3 --feature_iid 1 --label_iid 0 --alpha 1 >digit_fedpcl_filn_3bb_5u_a1.log
```
* To train on Office-10 with one backbone under the label shift non-IID setting:
```
python exps/federated_main.py --alg fedavg    --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 >office_fedavg_filn_1bb_5u_a1.log
python exps/federated_main.py --alg local     --dataset office --num_users 5 --rounds 100 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 >office_local_filn_1bb_5u_a1.log
python exps/federated_main.py --alg fedpcl    --dataset office --num_users 5 --rounds 100 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 >office_fedpcl_filn_1bb_5u_a1.log
```
* To train on Office-10 with three backbones under the label shift non-IID setting:
```
python exps/federated_main.py --alg fedavg    --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 1 --label_iid 0 --alpha 1 >office_fedavg_filn_3bb_5u_a1.log
python exps/federated_main.py --alg local     --dataset office --num_users 5 --rounds 100 --num_bb 3 --feature_iid 1 --label_iid 0 --alpha 1 >office_local_filn_3bb_5u_a1.log
python exps/federated_main.py --alg fedpcl    --dataset office --num_users 5 --rounds 60  --num_bb 3 --feature_iid 1 --label_iid 0 --alpha 1 >office_fedpcl_filn_3bb_5u_a1.log
```

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'digit'. Options: 'digit', 'office', 'domainnet'.
* ```--num_classes:```  Default: 10. 
* ```--alg:```      Default: 'fedpcl'. Options: 'fedpcl', 'fedavg', 'local'.
* ```--lr:```       Learning rate set to 0.001 by default.
* ```--local_bs:```  Local batch size set to 32 by default.
* ```--optimizer:```  The optimizer set to 'adam' by default.
* ```--model:```  Default: 'cnn'. Options: 'cnn', 'vit'.
* ```--num_bb:```     Default: 3. Options: 1, 3.
* ```--data_dir:```     Default: './data/'.
* ```--feature_iid:```     Default: 0. Default set to feature non-IID. Set to 1 for feature IID.
* ```--label_iid:```     Default: 1. Default set to label IID. Set to 0 for label non-IID.
* ```--alpha:```     Default: 1. The parameter of Dirichlet distribution that controls the non-IID level.

