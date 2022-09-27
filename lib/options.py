#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments
    parser.add_argument('--rounds', type=int, default=60, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--alg', type=str, default='fedpcl', help="algorithms")
    parser.add_argument('--train_ep', type=int, default=1, help="the number of local episodes: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size")
    parser.add_argument('--test_bs', type=int, default=32, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0, help='Adam weight decay (default: 0)')
    parser.add_argument('--device', default="cuda", type=str, help="cpu, cuda, or others")
    parser.add_argument('--gpu', default=0, type=int, help="index of gpu")
    parser.add_argument('--optimizer', type=str, default='adam', help="type of optimizer")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name') #cnn
    parser.add_argument('--num_bb', type=int, default=3, help='number of backbone')

    # data arguments
    parser.add_argument('--dataset', type=str, default='digit', help="name of dataset, e.g. digit")
    parser.add_argument('--percent', type=float, default=1, help="percentage of dataset to train")
    parser.add_argument('--data_dir', type=str, default='./data/', help="name of dataset, default: './data/'")
    parser.add_argument('--train_size', type=int, default=10, help="number of training samples in total")
    parser.add_argument('--test_size', type=int, default=100, help="num of test samples per dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--feature_iid', type=int, default=0, help='Default set to feature non-IID. Set to 1 for feature IID.')
    parser.add_argument('--label_iid', type=int, default=1, help='Default set to label non-IID. Set to 1 for label IID.')
    parser.add_argument('--test_ep', type=int, default=10, help="num of test episodes for evaluation")
    parser.add_argument('--save_protos', type=int, default=1, help="whether to save protos or not")

    # Local arguments
    parser.add_argument('--n_per_class', type=int, default=10, help="num of samples per class")
    parser.add_argument('--ld', type=float, default=0.5, help="weight of proto loss")
    parser.add_argument('--t', type=float, default=2, help="coefficient of local loss")
    parser.add_argument('--alpha', type=float, default=1, help="diri distribution parameter")

    # noise
    parser.add_argument('--add_noise_img', type=int, default=0, help="whether to add noise to images")
    parser.add_argument('--add_noise_proto', type=int, default=0, help="whether to add noise to images")
    parser.add_argument('--noise_type', type=str, default='gaussian', help="laplacian, gaussian, exponential")
    parser.add_argument('--perturb_coe', type=float, default=0.1, help="perturbation coefficient")
    parser.add_argument('--scale', type=float, default=0.05, help="noise distribution std")
    args = parser.parse_args()
    return args
