#!/bin/bash
# bash ./scripts/baseline.sh
echo script name: $0
echo $# arguments

#office
python exps/federated_main.py --alg fedavg    --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 >office_fedavg_filn_1bb_5u_a1.log
python exps/federated_main.py --alg local     --dataset office --num_users 5 --rounds 100 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 >office_local_filn_1bb_5u_a1.log
python exps/federated_main.py --alg fedpcl    --dataset office --num_users 5 --rounds 100 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 >office_fedpcl_filn_1bb_5u_a1.log

