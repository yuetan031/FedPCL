#!/bin/bash
# bash ./scripts/baseline.sh
echo script name: $0
echo $# arguments

#digit
python exps/federated_main.py --alg fedavg    --dataset digit --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 0 --alpha 1 >digit_fedavg_fnln_3bb_5u_a1.log
python exps/federated_main.py --alg local     --dataset digit --num_users 5 --rounds 100 --num_bb 3 --feature_iid 0 --label_iid 0 --alpha 1 >digit_local_fnln_3bb_5u_a1.log
python exps/federated_main.py --alg fedpcl    --dataset digit --num_users 5 --rounds 100 --num_bb 3 --feature_iid 0 --label_iid 0 --alpha 1 >digit_fedpcl_fnln_3bb_5u_a1.log
python exps/federated_main.py --alg fedpcl    --model vit --lr 0.03 --optimizer sgd --dataset digit --num_users 5 --rounds 100 --num_bb 3 --feature_iid 0 --label_iid 0 --alpha 1 >digit_fedpcl_fnln_3bb_5u_vit.log

