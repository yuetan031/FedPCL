#!/bin/bash
# bash ./scripts/baseline.sh
echo script name: $0
echo $# arguments

#domainnet
python exps/federated_main.py --alg local    --dataset domainnet --num_users 5 --rounds 100 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 >domainnet_local_filn_1bb_5u_a1.log
python exps/federated_main.py --alg fedavg   --dataset domainnet --num_users 5 --rounds 300 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 >domainnet_fedavg_filn_1bb_5u_a1_300r.log
python exps/federated_main.py --alg fedpcl   --dataset domainnet --num_users 5 --rounds 200 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 >domainnet_fedpcl_filn_1bb_5u_a1.log

