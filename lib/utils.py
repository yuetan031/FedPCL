#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_lt
from sampling import femnist_iid, femnist_noniid, femnist_noniid_unequal, femnist_noniid_lt
import numpy as np
import data_utils
from data_utils import TwoCropTransform
import seaborn as sns
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    print('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def partition_data(dataset, iid, num_users, alpha, args):
    if dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(args.data_dir, train=True, download=False,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(args.data_dir, train=False, download=False,
                                      transform=apply_transform)

        X_train, y_train = train_dataset.data[0:args.train_size], train_dataset.targets[0:args.train_size]
        n_train = X_train.shape[0]
    else:
        raise ValueError("Unknown dataset : {:}".format(dataset))

    iid = 0
    if iid:
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, num_users)
    else:
        min_size = 0
        K = args.num_classes
        N = y_train.shape[0]
        user_groups = {}
        user_groups_test = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(num_users)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            user_groups[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, user_groups)

    np.random.shuffle(idx_k)
    test_size = args.test_size
    for i in range(num_users):
        user_groups_test[i] = np.arange(i * test_size, (i + 1) * test_size)

    return train_dataset, test_dataset, y_train, user_groups, user_groups_test, traindata_cls_counts

def add_noise_img(img, scale, perturb_coe, noise_type):
    scale_scalar = scale
    scale = torch.full(size=img.shape, fill_value=scale_scalar, dtype=torch.float32)
    if noise_type == "gaussian":
        dist = torch.distributions.normal.Normal(0, scale)
    elif noise_type == "laplacian":
        dist = torch.distributions.laplace.Laplace(0, scale)
    elif noise_type == "exponential":
        rate = 1 / scale
        dist = torch.distributions.exponential.Exponential(rate)
    else:
        dist = torch.distributions.normal.Normal(0, scale)
    noise = dist.sample()

    return img * (1 - perturb_coe) + noise

def add_noise_proto(device, local_protos, scale, perturb_coe, noise_type):
    scale_scalar = scale
    for label in local_protos.keys():
        scale = torch.full(size=local_protos[label].shape, fill_value=scale_scalar, dtype=torch.float32)
        if noise_type == "gaussian":
            dist = torch.distributions.normal.Normal(0, scale)
        elif noise_type == "laplacian":
            dist = torch.distributions.laplace.Laplace(0, scale)
        elif noise_type == "exponential":
            rate = 1 / scale
            dist = torch.distributions.exponential.Exponential(rate)
        else:
            dist = torch.distributions.normal.Normal(0, scale)
        noise = dist.sample().to(device)
        local_protos[label] = local_protos[label] * (1 - perturb_coe) + noise

    return local_protos

def prepare_data_digits(num_users, args):
    # Prepare digit (feature noniid, label iid)
    if args.model == 'cnn':
        transform_mnist = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_svhn = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_usps = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_synth = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_mnistm = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif args.model == 'vit':
        transform_mnist = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_svhn = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_usps = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_synth = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_mnistm = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    data_root = './data/'

    # MNIST
    mnist_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + "digit/MNIST", channels=1, train=True, transform=TwoCropTransform(transform_mnist))
    mnist_testset = data_utils.DigitsDataset(args=args, data_path=data_root + "digit/MNIST", channels=1, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/SVHN', channels=3, train=True, transform=TwoCropTransform(transform_svhn))
    svhn_testset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/SVHN', channels=3, train=False, transform=transform_svhn)

    # Synth Digits
    synth_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/SynthDigits/', channels=3, train=True, transform=TwoCropTransform(transform_synth))
    synth_testset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/SynthDigits/', channels=3, train=False, transform=transform_synth)

    # USPS
    usps_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/USPS', channels=1, train=True, transform=TwoCropTransform(transform_usps))
    usps_testset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/USPS', channels=1, train=False, transform=transform_usps)

    # MNIST-M
    mnistm_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=True, transform=TwoCropTransform(transform_mnistm))
    mnistm_testset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=False, transform=transform_mnistm)

    train_dataset_list = [mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset]
    test_dataset_list = [mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset]

    K = args.num_classes
    idx_batch_train = [[] for _ in range(num_users)]
    user_groups = {}
    for i in range(num_users):
        y_train = train_dataset_list[i].labels
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_batch_train[i].extend(idx_k[0:10].tolist())
        user_groups[i] = idx_batch_train[i]

    # test idx
    user_groups_test = {}
    idx_batch_test = [[] for _ in range(num_users)]
    for i in range(num_users):
        y_test = test_dataset_list[i].labels
        for k in range(K):
            idx_k = np.where(y_test == k)[0]
            # idx_batch_test[i].extend(idx_k[start : end].tolist())
            idx_batch_test[i].extend(idx_k[0:200].tolist())
        user_groups_test[i] = idx_batch_test[i]

    visualize_data_dist_fnli(args, [], K, num_users)

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

def prepare_data_digits_noniid(num_users, args):
    # Prepare digit (feature noniid, label iid)
    if args.model == 'cnn':
        transform_mnist = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_svhn = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_usps = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_synth = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_mnistm = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif args.model == 'vit':
        transform_mnist = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_svhn = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_usps = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_synth = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_mnistm = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    data_root = args.data_dir

    # MNIST
    mnist_trainset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/MNIST', channels=1, train=True, transform=TwoCropTransform(transform_mnist))
    mnist_testset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/MNIST', channels=1, train=False, transform=transform_mnist)

    # SVHN
    svhn_trainset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/SVHN', channels=3, train=True, transform=TwoCropTransform(transform_svhn))
    svhn_testset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/SVHN', channels=3, train=False, transform=transform_svhn)

    # USPS
    usps_trainset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/USPS', channels=1, train=True, transform=TwoCropTransform(transform_usps))
    usps_testset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/USPS', channels=1, train=False, transform=transform_usps)

    # Synth Digits
    synth_trainset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/SynthDigits/', channels=3, train=True, transform=TwoCropTransform(transform_synth))
    synth_testset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/SynthDigits/', channels=3, train=False, transform=transform_synth)

    # MNIST-M
    mnistm_trainset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/MNIST_M/', channels=3, train=True, transform=TwoCropTransform(transform_mnistm))
    mnistm_testset = data_utils.DigitsDataset(args=args, data_path=data_root+'digit/MNIST_M/', channels=3, train=False, transform=transform_mnistm)

    train_dataset_list = [mnist_trainset, svhn_trainset, usps_trainset, synth_trainset, mnistm_trainset]
    test_dataset_list = [mnist_testset, svhn_testset, usps_testset, synth_testset, mnistm_testset]

    # generate train idx
    idx_batch_train = [[] for _ in range(num_users)]
    user_groups = {}
    K = args.num_classes
    df = np.zeros([num_users, K])
    for k in range(K):
        proportions = np.random.dirichlet(np.repeat(args.alpha, num_users))
        proportions = proportions / proportions.sum()
        proportions = ((proportions) * (num_users*10)).astype(int)
        for i in range(num_users):
            y_train = train_dataset_list[i].labels
            idx_k = np.where(y_train == k)[0]
            idx_batch_train[i].extend(idx_k[0:proportions[i]].tolist())

        j = 0
        for idx_j in idx_batch_train:
            if k != 0:
                df[j, k] = int(len(idx_j))
            else:
                df[j, k] = int(len(idx_j))
            j += 1

    for i in range(num_users):
        user_groups[i] = idx_batch_train[i]

    # generate test idx
    user_groups_test = {}
    idx_batch_test = [[] for _ in range(num_users)]
    for i in range(num_users):
        y_test = test_dataset_list[i].labels
        for k in range(K):
            idx_k = np.where(y_test == k)[0]
            # idx_batch_test[i].extend(idx_k[start : end].tolist())
            idx_batch_test[i].extend(idx_k[0:100].tolist())
        user_groups_test[i] = idx_batch_test[i]

    # visualize data distribution
    visualize_data_dist_fnln(args, df, K, num_users)

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

def prepare_data_mnistm_iid(num_users, args):
    # Prepare digit (feature iid, label iid)
    transform_synth = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Synth Digits
    synth_trainset = data_utils.DigitsDataset(args=args, data_path='./data/digit/MNIST_M/', channels=3, train=False, transform=TwoCropTransform(transform_synth))
    synth_testset = data_utils.DigitsDataset(args=args, data_path='./data/digit/MNIST_M/', channels=3, train=False, transform=transform_synth)

    train_dataset_list = []
    test_dataset_list = []
    for _ in range(num_users):
        train_dataset_list.append(synth_trainset)
        test_dataset_list.append(synth_testset)

    K = args.num_classes
    idx_batch_train = [[] for _ in range(num_users)]
    user_groups = {}
    n_sample_per_class = 10
    for i in range(num_users):
        y_train = train_dataset_list[i].labels
        start = i * n_sample_per_class
        end = start + n_sample_per_class
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_batch_train[i].extend(idx_k[start : end].tolist())
        user_groups[i] = idx_batch_train[i]

    # test idx
    user_groups_test = {}
    idx_batch_test = [[] for _ in range(num_users)]
    n_sample_per_class_test = 100
    for i in range(num_users):
        y_test = test_dataset_list[i].labels
        start = i * n_sample_per_class_test
        end = start + n_sample_per_class_test
        for k in range(K):
            idx_k = np.where(y_test == k)[0]
            idx_batch_test[i].extend(idx_k[start : end].tolist())
        user_groups_test[i] = idx_batch_test[i]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

def prepare_data_mnistm_noniid(num_users, args):
    data_root = args.data_dir

    if args.model == 'cnn':
        transform_mnistm = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Synth Digits
        mnistm_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=False, transform=TwoCropTransform(transform_mnistm))
        mnistm_testset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=False, transform=transform_mnistm)

    elif args.model == 'vit':
        transform_mnistm_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        transform_mnistm_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # Synth Digits
        mnistm_trainset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=False, transform=TwoCropTransform(transform_mnistm_train))
        mnistm_testset = data_utils.DigitsDataset(args=args, data_path=data_root + 'digit/MNIST_M/', channels=3, train=False, transform=transform_mnistm_test)

    train_dataset_list = []
    test_dataset_list = []
    for _ in range(num_users):
        train_dataset_list.append(mnistm_trainset)
        test_dataset_list.append(mnistm_testset)

    # generate train idx and test idx
    K = args.num_classes
    idx_batch = [[] for _ in range(num_users)]
    y = mnistm_trainset.labels
    N = y.shape[0]
    df = np.zeros([num_users, K])
    for k in range(K):
        idx_k = np.where(y == k)[0]
        if num_users ==5 or num_users == 10:
            idx_k = idx_k[0:110*num_users]
        elif num_users == 20:
            idx_k = idx_k[0:55 * num_users]
        elif num_users == 40:
            idx_k = idx_k[0:30 * num_users]
        elif num_users == 80:
            idx_k = idx_k[0:15 * num_users]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(args.alpha, num_users))
        proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

        j = 0
        for idx_j in idx_batch:
            if k != 0:
                df[j, k] = int(len(idx_j))
            else:
                df[j, k] = int(len(idx_j))
            j += 1

    user_groups = {}
    user_groups_test = {}
    for i in range(num_users):
        np.random.shuffle(idx_batch[i])
        num_samples = len(idx_batch[i])
        if num_users == 5 or num_users == 10:
            train_len = int(num_samples/11)
        elif num_users == 20:
            train_len = int(num_samples / 5.5)
        elif num_users == 40:
            train_len = int(num_samples / 3)
        elif num_users == 80:
            train_len = int(num_samples / 1.5)
        user_groups[i] = idx_batch[i][:train_len]
        user_groups_test[i] = idx_batch[i][train_len:]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

def prepare_data_office(num_users, args):
    # Prepare office (feature noniid, label iid)
    if args.model == 'cnn':
        transform_office = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
        ])
    elif args.model == 'vit':
        transform_office = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

    data_base_path = "./data/office_caltech_10/"

    # amazon
    amazon_trainset = data_utils.OfficeDataset(data_base_path, 'amazon', transform=TwoCropTransform(transform_office))
    amazon_testset = data_utils.OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)

    # caltech
    caltech_trainset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=TwoCropTransform(transform_office))
    caltech_testset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)

    # dslr
    dslr_trainset = data_utils.OfficeDataset(data_base_path, 'dslr', transform=TwoCropTransform(transform_office))
    dslr_testset = data_utils.OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)

    # webcam
    webcam_trainset = data_utils.OfficeDataset(data_base_path, 'webcam', transform=TwoCropTransform(transform_office))
    webcam_testset = data_utils.OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    train_dataset_list = [amazon_trainset, caltech_trainset, dslr_trainset, webcam_trainset]
    test_dataset_list = [amazon_testset, caltech_testset, dslr_testset, webcam_testset]

    if args.label_iid:
        K = args.num_classes
        idx_batch_train = [[] for _ in range(num_users)]
        user_groups = {}
        for i in range(num_users):
            ds_idx = ds_idx_list[i]
            y_train = train_dataset_list[ds_idx].labels
            y_train = torch.FloatTensor(y_train)
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                idx_batch_train[i].extend(idx_k[0:50].tolist())
            user_groups[i] = idx_batch_train[i]
        # test idx
        user_groups_test = {}
        idx_batch_test = [[] for _ in range(num_users)]
        for i in range(num_users):
            user_groups_test[i] = []
            y_test = test_dataset_list[i].labels
            y_test = torch.FloatTensor(y_test)
            idx_batch_test[i] = []
            for k in range(K):
                idx_k = np.where(y_test == k)[0]
                idx_batch_test[i].extend(idx_k[0:100].tolist())
            user_groups_test[i] = idx_batch_test[i]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

def prepare_data_office_noniid(num_users, args):
    # Prepare office (feature noniid, label noniid)
    if args.model == 'cnn' or args.model == 'other':
        transform_office = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
        ])
    elif args.model == 'vit':
        transform_office = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

    data_base_path = "./data/office/"

    # amazon
    amazon_trainset = data_utils.OfficeDataset(data_base_path, 'amazon', transform=TwoCropTransform(transform_office))
    amazon_testset = data_utils.OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)

    # caltech
    caltech_trainset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=TwoCropTransform(transform_office))
    caltech_testset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)

    # dslr
    dslr_trainset = data_utils.OfficeDataset(data_base_path, 'dslr', transform=TwoCropTransform(transform_office))
    dslr_testset = data_utils.OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)

    # webcam
    webcam_trainset = data_utils.OfficeDataset(data_base_path, 'webcam', transform=TwoCropTransform(transform_office))
    webcam_testset = data_utils.OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    train_dataset_list = [amazon_trainset, caltech_trainset, dslr_trainset, webcam_trainset]
    test_dataset_list = [amazon_testset, caltech_testset, dslr_testset, webcam_testset]

    # generate train idx
    idx_batch_train = [[] for _ in range(num_users)]
    user_groups = {}
    K = args.num_classes
    for k in range(K):
        proportions = np.random.dirichlet(np.repeat(args.alpha, num_users))
        proportions = proportions / proportions.sum()
        proportions = ((proportions) * (num_users*10)).astype(int)
        for i in range(num_users):
            y_train = train_dataset_list[i].labels
            y_train = torch.FloatTensor(y_train)
            idx_k = np.where(y_train == k)[0]
            idx_batch_train[i].extend(idx_k[0:proportions[i]].tolist())

    for i in range(num_users):
        user_groups[i] = idx_batch_train[i]

    # generate test idx
    user_groups_test = {}
    idx_batch_test = [[] for _ in range(num_users)]
    for i in range(num_users):
        y_test = test_dataset_list[i].labels
        y_test = torch.FloatTensor(y_test)
        for k in range(K):
            idx_k = np.where(y_test == k)[0]
            idx_batch_test[i].extend(idx_k[0:100].tolist())
        user_groups_test[i] = idx_batch_test[i]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

def prepare_data_caltech_noniid(num_users, args):
    # Prepare data
    transform_office = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
    ])

    data_root = args.data_dir

    # caltech
    caltech_trainset = data_utils.OfficeDataset(data_root+'office/', 'caltech', transform=TwoCropTransform(transform_office), train=True)
    caltech_testset = data_utils.OfficeDataset(data_root+'office/', 'caltech', transform=transform_test, train=True)

    train_dataset_list = []
    test_dataset_list = []
    for _ in range(num_users):
        train_dataset_list.append(caltech_trainset)
        test_dataset_list.append(caltech_testset)

    # generate train idx and test idx
    K = args.num_classes
    idx_batch = [[] for _ in range(num_users)]
    y = np.array(caltech_trainset.labels)
    N = y.shape[0]
    df = np.zeros([num_users, K])
    for k in range(K):
        idx_k = np.where(y == k)[0]
        idx_k = idx_k[0:30 * num_users]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(args.alpha, num_users))
        proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        j = 0
        for idx_j in idx_batch:
            if k != 0:
                df[j, k] = int(len(idx_j))
            else:
                df[j, k] = int(len(idx_j))
            j += 1

    user_groups = {}
    user_groups_test = {}
    for i in range(num_users):
        np.random.shuffle(idx_batch[i])
        num_samples = len(idx_batch[i])
        train_len = int(num_samples / 2)
        user_groups[i] = idx_batch[i][:train_len]
        user_groups_test[i] = idx_batch[i][train_len:]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

def prepare_data_caltech_iid(num_users, args):
    # Prepare data
    transform_office = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
    ])

    data_base_path = "./data/office_caltech_10/"

    # caltech
    caltech_trainset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=TwoCropTransform(transform_office), train=True)
    caltech_testset = data_utils.OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=True)

    train_dataset_list = []
    test_dataset_list = []
    for _ in range(num_users):
        train_dataset_list.append(caltech_trainset)
        test_dataset_list.append(caltech_testset)

    # generate train idx and test idx
    K = args.num_classes
    idx_batch = [[] for _ in range(num_users)]
    y = np.array(caltech_trainset.labels)
    N = y.shape[0]
    df = np.zeros([num_users, K])
    for k in range(K):
        idx_k = np.where(y == k)[0]
        idx_k = idx_k[0:30 * num_users]
        np.random.shuffle(idx_k)
        proportions = np.array([0.2,0.2,0.2,0.2,0.2])
        proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        j = 0
        for idx_j in idx_batch:
            if k != 0:
                df[j, k] = int(len(idx_j))
            else:
                df[j, k] = int(len(idx_j))
            j += 1

    user_groups = {}
    user_groups_test = {}
    for i in range(num_users):
        np.random.shuffle(idx_batch[i])
        num_samples = len(idx_batch[i])
        train_len = int(num_samples / 2)
        user_groups[i] = idx_batch[i][:train_len]
        user_groups_test[i] = idx_batch[i][train_len:]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

def prepare_data_domainnet(num_users, args):
    # Prepare data
    if args.model == 'cnn':
        transform_train = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ])
    elif args.model == 'vit':
        transform_train = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30, 30)),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
        ])

    data_base_path = "./data/domainnet/"

    # clipart
    clipart_trainset = data_utils.DomainNetDataset(data_base_path, 'clipart', transform=TwoCropTransform(transform_train))
    clipart_testset = data_utils.DomainNetDataset(data_base_path, 'clipart', transform=transform_test, train=False)
    # infograph
    infograph_trainset = data_utils.DomainNetDataset(data_base_path, 'infograph', transform=TwoCropTransform(transform_train))
    infograph_testset = data_utils.DomainNetDataset(data_base_path, 'infograph', transform=transform_test, train=False)
    # painting
    painting_trainset = data_utils.DomainNetDataset(data_base_path, 'painting', transform=TwoCropTransform(transform_train))
    painting_testset = data_utils.DomainNetDataset(data_base_path, 'painting', transform=transform_test, train=False)
    # quickdraw
    quickdraw_trainset = data_utils.DomainNetDataset(data_base_path, 'quickdraw', transform=TwoCropTransform(transform_train))
    quickdraw_testset = data_utils.DomainNetDataset(data_base_path, 'quickdraw', transform=transform_test, train=False)
    # real
    real_trainset = data_utils.DomainNetDataset(data_base_path, 'real', transform=TwoCropTransform(transform_train))
    real_testset = data_utils.DomainNetDataset(data_base_path, 'real', transform=transform_test, train=False)
    # sketch
    sketch_trainset = data_utils.DomainNetDataset(data_base_path, 'sketch', transform=TwoCropTransform(transform_train))
    sketch_testset = data_utils.DomainNetDataset(data_base_path, 'sketch', transform=transform_test, train=False)

    train_dataset_list = [clipart_trainset, infograph_trainset, painting_trainset, quickdraw_trainset,
                     real_trainset, sketch_trainset]
    test_dataset_list = [clipart_testset, infograph_testset, painting_testset, quickdraw_testset,
                    real_testset, sketch_testset]

    K = args.num_classes
    idx_batch_train = [[] for _ in range(num_users)]
    user_groups = {}
    for i in range(num_users):
        ds_idx = ds_idx_list[i]
        y_train = torch.Tensor(train_dataset_list[ds_idx].labels)
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_batch_train[i].extend(idx_k[:].tolist())
        user_groups[i] = list(range(105))

    # test idx
    user_groups_test = {}
    idx_batch_test = [[] for _ in range(num_users)]
    for i in range(num_users):
        y_test = test_dataset_list[i].labels
        # for k in range(K):
        #     idx_k = np.where(y_test == k)[0]
        #     idx_batch_test[i].extend(idx_k[:].tolist())
        user_groups_test[i] = list(range(len(y_test)))

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

def prepare_data_domainnet_noniid(num_users, args):
    # Prepare data
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    data_base_path = "./data/domainnet/"

    # clipart
    clipart_trainset = data_utils.DomainNetDataset(data_base_path, 'clipart', transform=TwoCropTransform(transform_train))
    clipart_testset = data_utils.DomainNetDataset(data_base_path, 'clipart', transform=transform_test, train=False)
    # infograph
    infograph_trainset = data_utils.DomainNetDataset(data_base_path, 'infograph', transform=TwoCropTransform(transform_train))
    infograph_testset = data_utils.DomainNetDataset(data_base_path, 'infograph', transform=transform_test, train=False)
    # painting
    painting_trainset = data_utils.DomainNetDataset(data_base_path, 'painting', transform=TwoCropTransform(transform_train))
    painting_testset = data_utils.DomainNetDataset(data_base_path, 'painting', transform=transform_test, train=False)
    # quickdraw
    quickdraw_trainset = data_utils.DomainNetDataset(data_base_path, 'quickdraw', transform=TwoCropTransform(transform_train))
    quickdraw_testset = data_utils.DomainNetDataset(data_base_path, 'quickdraw', transform=transform_test, train=False)
    # real
    real_trainset = data_utils.DomainNetDataset(data_base_path, 'real', transform=TwoCropTransform(transform_train))
    real_testset = data_utils.DomainNetDataset(data_base_path, 'real', transform=transform_test, train=False)
    # sketch
    sketch_trainset = data_utils.DomainNetDataset(data_base_path, 'sketch', transform=TwoCropTransform(transform_train))
    sketch_testset = data_utils.DomainNetDataset(data_base_path, 'sketch', transform=transform_test, train=False)

    train_dataset_list = [clipart_trainset, infograph_trainset, painting_trainset, quickdraw_trainset, real_trainset, sketch_trainset]
    test_dataset_list = [clipart_testset, infograph_testset, painting_testset, quickdraw_testset, real_testset, sketch_testset]

    # generate train idx
    idx_batch_train = [[] for _ in range(num_users)]
    user_groups = {}
    K = args.num_classes
    for k in range(K):
        proportions = np.random.dirichlet(np.repeat(args.alpha, num_users))
        proportions = proportions / proportions.sum()
        proportions = ((proportions) * (num_users * 20)).astype(int)
        for i in range(num_users):
            y_train = train_dataset_list[i].labels
            y_train = torch.FloatTensor(y_train)
            idx_k = np.where(y_train == k)[0]
            idx_batch_train[i].extend(idx_k[0:proportions[i]].tolist())

    for i in range(num_users):
        user_groups[i] = idx_batch_train[i]

    # generate test idx
    user_groups_test = {}
    idx_batch_test = [[] for _ in range(num_users)]
    for i in range(num_users):
        y_test = test_dataset_list[i].labels
        y_test = torch.FloatTensor(y_test)
        # n_sample_per_class_test = args.test_size
        # start = i * n_sample_per_class_test
        # end = start + n_sample_per_class_test
        for k in range(K):
            idx_k = np.where(y_test == k)[0]
            # idx_batch_test[i].extend(idx_k[start : end].tolist())
            idx_batch_test[i].extend(idx_k[0:100].tolist())
        user_groups_test[i] = idx_batch_test[i]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

def prepare_data_real_noniid(num_users, args):
    # Prepare data
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    data_root = args.data_dir

    # real
    real_trainset = data_utils.DomainNetDataset(data_root+'domainnet/', 'real', transform=TwoCropTransform(transform_train), train=True)
    real_testset = data_utils.DomainNetDataset(data_root+'domainnet/', 'real', transform=transform_test, train=True)

    train_dataset_list = []
    test_dataset_list = []
    for _ in range(num_users):
        train_dataset_list.append(real_trainset)
        test_dataset_list.append(real_testset)

    # generate train idx and test idx
    K = args.num_classes
    idx_batch = [[] for _ in range(num_users)]
    y = np.array(real_trainset.labels)
    N = y.shape[0]
    df = np.zeros([num_users, K])
    for k in range(K):
        idx_k = np.where(y == k)[0]
        print("len(idx_k)::",len(idx_k))
        idx_k = idx_k[0:40 * num_users]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(args.alpha, num_users))
        proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        # min_size = min([len(idx_j) for idx_j in idx_batch])
        j = 0
        for idx_j in idx_batch:
            if k != 0:
                df[j, k] = int(len(idx_j))
            else:
                df[j, k] = int(len(idx_j))
            j += 1

    user_groups = {}
    user_groups_test = {}
    for i in range(num_users):
        np.random.shuffle(idx_batch[i])
        num_samples = len(idx_batch[i])
        train_len = int(num_samples / 4)
        user_groups[i] = idx_batch[i][:train_len]
        user_groups_test[i] = idx_batch[i][train_len:]

    return train_dataset_list, test_dataset_list, user_groups, user_groups_test

def visualize_data_dist_filn(args, df, K, num_users):

    for k in range(K - 1, 0, -1):
        for j in range(df.shape[0]):
            df[j, k] = df[j, k] - df[j, k - 1]
    df_1 = np.zeros([num_users * K, 3])

    for j in range(num_users):
        for k in range(K):
            for i in range(3):
                df_1[j * K + k, 0] = int(j)
                df_1[j * K + k, 1] = int(k)
                df_1[j * K + k, 2] = int(df[j, k])

    # df = pd.DataFrame(data=df_test_1, index=["row1", "row2"], columns=["column1", "column2"])
    sns.set_theme(style="darkgrid")

    sns.set(font_scale=1.5) ####

    corr_mat = pd.DataFrame(data=df_1, columns=['Client ID', 'Class', '# of samples'])

    g = sns.relplot(
        data=corr_mat,
        x='Client ID', y='Class', hue='# of samples', size='# of samples',
        palette="vlag", hue_norm=(-1, 1), edgecolor=".7", sizes=(50, 250),
    )

    g.set(xlabel="Client ID", ylabel="Class", xticks=np.arange(0, num_users, 1), yticks=np.arange(0, K, 1))

    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")

    g.savefig('./noniid_train_'+ str(args.num_classes) + 'c_' +str(num_users)+ 'u_' +'a' +str(args.alpha)+'_f'+str(args.feature_iid)+'_l'+str(args.label_iid)+'.pdf', format='pdf')

def visualize_data_dist_fnli(args, df, K, num_users):
    df_1 = np.zeros([num_users * K, 3],dtype=int)
    for j in range(num_users):
        for k in range(K):
            for i in range(3):
                df_1[j * K + k, 0] = int(j)
                df_1[j * K + k, 1] = int(k)
                df_1[j * K + k, 2] = int(210)

    # df = pd.DataFrame(data=df_test_1, index=["row1", "row2"], columns=["column1", "column2"])
    sns.set_theme(style="darkgrid")

    # # Load the brain networks dataset, select subset, and collapse the multi-index
    # df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
    #
    # used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
    # used_columns = (df.columns
    #                 .get_level_values("network")
    #                 .astype(int)
    #                 .isin(used_networks))
    # df = df.loc[:, used_columns]
    #
    # df.columns = df.columns.map("-".join)

    # Compute a correlation matrix and convert to long-form
    # corr_mat = df.corr().stack().reset_index(name="correlation")
    sns.set(font_scale=1.5)  ####
    corr_mat = pd.DataFrame(data=df_1, columns=['Client ID', 'Class', '# of samples'])

    # Draw each cell as a scatter point with varying size and color
    # g = sns.relplot(
    #     data=corr_mat,
    #     x='client ID', y='class', hue='num of samples', size='num of samples',
    #     palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    #     height=10, sizes=(50, 250), size_norm=(0, 400),   palette="vlag",hue_norm=(-1, 1),
    # )
    g = sns.relplot(
        data=corr_mat,
        x='Client ID', y='Class', hue='Client ID', size='# of samples',
        palette="rocket", edgecolor=".7", sizes=(200, 250),
    )

    # Tweak the figure to finalize
    g.set(xlabel="Client ID", ylabel="Class", xticks=[0,1,2,3,4], yticks=np.arange(0, K, 1))
    plt.legend(labels=['','','MNIST','SVHN','USPS','Synth','MNIST-M'],bbox_to_anchor=(0.9,0.52))
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")
    g.savefig('./noniid_train_'+ str(args.num_classes) + 'c_' +str(num_users)+ 'u_' +'a' +str(args.alpha)+'_f'+str(args.feature_iid)+'_l'+str(args.label_iid)+'.pdf', format='pdf')

def visualize_data_dist_fnln(args, df, K, num_users):
    for k in range(K - 1, 0, -1):
        for j in range(df.shape[0]):
            df[j, k] = df[j, k] - df[j, k - 1]
    df_1 = np.zeros([num_users * K, 3], dtype=int)

    for j in range(num_users):
        for k in range(K):
            for i in range(3):
                df_1[j * K + k, 0] = int(j)
                df_1[j * K + k, 1] = int(k)
                df_1[j * K + k, 2] = int(df[j, k])*11

    # df = pd.DataFrame(data=df_test_1, index=["row1", "row2"], columns=["column1", "column2"])
    sns.set_theme(style="darkgrid")

    # # Load the brain networks dataset, select subset, and collapse the multi-index
    # df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
    #
    # used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
    # used_columns = (df.columns
    #                 .get_level_values("network")
    #                 .astype(int)
    #                 .isin(used_networks))
    # df = df.loc[:, used_columns]
    #
    # df.columns = df.columns.map("-".join)

    # Compute a correlation matrix and convert to long-form
    # corr_mat = df.corr().stack().reset_index(name="correlation")
    sns.set(font_scale=1.5)  ####
    corr_mat = pd.DataFrame(data=df_1, columns=['Client ID', 'Class', '# of samples'])

    # Draw each cell as a scatter point with varying size and color
    # g = sns.relplot(
    #     data=corr_mat,
    #     x='client ID', y='class', hue='num of samples', size='num of samples',
    #     palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    #     height=10, sizes=(50, 250), size_norm=(0, 400),   palette="vlag",hue_norm=(-1, 1),
    # )
    g = sns.relplot(
        data=corr_mat, x='Client ID', y='Class', hue='Client ID', size='# of samples',
        palette="rocket", edgecolor=".7", sizes=(50, 250),
    )

    # Tweak the figure to finalize
    g.set(xlabel="Client ID", ylabel="Class", xticks=[0,1,2,3,4], yticks=np.arange(0, K, 1))
    # plt.legend(labels=['','','MNIST','SVHN','USPS','Synth','MNIST-M'],bbox_to_anchor=(0.9,0.52))
    # g.despine(left=True, bottom=True)
    # g.ax.margins(.02)
    # for label in g.ax.get_xticklabels():
    #     label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")
    # g.fig.set_figwidth(5)
    # g.fig.set_figheight(3)
    g.savefig('./noniid_train_'+ str(args.num_classes) + 'c_' +str(num_users)+ 'u_' +'a' +str(args.alpha)+'_f'+str(args.feature_iid)+'_l'+str(args.label_iid)+'.pdf', format='pdf')

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[0][key] += w[i][key]
        w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg

def agg_func(protos):
    """
    Average the protos for each local user
    """
    agg_protos = {}
    for [label, proto_list] in protos.items():
        proto = torch.stack(proto_list)
        agg_protos[label] = torch.mean(proto, dim=0).data

    return agg_protos

def recon(args, backbone_list, model):
    device = args.device
    model.eval()
    model.requires_grad_(False)

    images = np.load('./images.npy', allow_pickle=True)
    labels = np.load('./labels.npy', allow_pickle=True)
    features = np.load('./save/features.npy', allow_pickle=True)
    protos = np.load('./protos_4.npy', allow_pickle=True)

    images = torch.from_numpy(images)
    features = torch.from_numpy(features).to(device)
    protos = torch.from_numpy(protos).to(device)

    dummy_data = torch.randn(images.size()).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([dummy_data], lr=0.001, weight_decay=1e-4)
    for iter in range(300):

        optimizer.zero_grad()
        for i in range(len(backbone_list)):
            backbone = backbone_list[i]
            if i == 0:
                reps = backbone(dummy_data)
            else:
                reps = torch.cat((reps, backbone(dummy_data)), 1)

        _, dummy_features = model(reps)
        loss_mse = nn.MSELoss()
        dummy_loss = loss_mse(dummy_features, features)
        print('Iteration : {}\tLoss: {:.3f}'.format(iter, dummy_loss.item()))
        dummy_loss.backward()

        optimizer.step()

    tt = transforms.ToPILImage()
    tp = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ])
    plt.imshow(tt(dummy_data[0,:].cpu()))
    plt.axis('off')
    plt.show()

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')

    print('    Federated parameters:')
    if args.label_iid:
        print('   Label IID')
    else:
        print('   Label Non-IID')
    if args.feature_iid:
        print('   Feature IID')
    else:
        print('   Feature Non-IID')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.train_ep}\n')
    return
