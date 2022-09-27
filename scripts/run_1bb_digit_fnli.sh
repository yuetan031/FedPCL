#!/bin/bash
# bash ./scripts/baseline.sh
echo script name: $0
echo $# arguments


#digit
python exps/federated_main.py --alg fedavg    --dataset digit --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 >digit_fedavg_fnli_1bb_5u.log
python exps/federated_main.py --alg local     --dataset digit --num_users 5 --rounds 100 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 >digit_local_fnli_1bb_5u.log
python exps/federated_main.py --alg fedpcl    --dataset digit --num_users 5 --rounds 100 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 >digit_fedpcl_fnli_1bb_5u.log

