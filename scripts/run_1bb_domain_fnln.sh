#!/bin/bash
# bash ./scripts/baseline.sh
echo script name: $0
echo $# arguments

#domainnet
python exps/federated_main.py --alg local     --dataset domainnet --num_users 6 --rounds 100 --num_bb 1 --feature_iid 0 --label_iid 0 --alpha 1 >domainnet_local_fnln_1bb_6u_a1.log
python exps/federated_main.py --alg fedavg    --dataset domainnet --num_users 6 --rounds 300 --num_bb 1 --feature_iid 0 --label_iid 0 --alpha 1 >domainnet_fedavg_fnln_1bb_6u_a1_300r.log
python exps/federated_main.py --alg fedpcl    --dataset domainnet --num_users 6 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 0 --alpha 1 >domainnet_fedpcl_fnln_1bb_6u_a1.log

