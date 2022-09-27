#!/bin/bash
# bash ./scripts/baseline.sh
echo script name: $0
echo $# arguments

#office
python exps/federated_main.py --alg fedavg    --dataset office --num_users 4 --rounds 200 --num_bb 1 --local_bs 16 --feature_iid 0 --label_iid 0 --alpha 1 >office_fedavg_fnln_1bb_5u_a1.log
python exps/federated_main.py --alg local     --dataset office --num_users 4 --rounds 100 --num_bb 1 --local_bs 16 --feature_iid 0 --label_iid 0 --alpha 1 >office_local_fnln_1bb_5u_a1.log
python exps/federated_main.py --alg fedpcl    --dataset office --num_users 4 --rounds 100 --num_bb 1 --local_bs 16 --feature_iid 0 --label_iid 0 --alpha 1 >office_fedpcl_fnln_1bb_5u_a1.log

