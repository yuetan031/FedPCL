#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os, sys, copy
import time, random
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from pathlib import Path
from datetime import datetime

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))


from models.resnet import resnet18
from models.vision_transformer import vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
from options import args_parser
from update import LocalUpdate, LocalTest
from models.models import ProjandDeci
from models.multibackbone import alexnet, vgg11, mlp_m
from utils import add_noise_proto, prepare_data_real_noniid, prepare_data_domainnet_noniid, prepare_data_office_noniid, prepare_data_digits_noniid, prepare_data_caltech_noniid, prepare_data_mnistm_noniid, average_weights, exp_details, proto_aggregation, agg_func, prepare_data_digits, prepare_data_office, prepare_data_domainnet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def Local(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    idxs_users = np.arange(args.num_users)
    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round} |\n')
        for idx in idxs_users:
            local_node = LocalUpdate(args=args, dataset=train_dataset_list[idx],idxs=user_groups[idx])
            w, loss = local_node.update_weights(idx, backbone_list=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss, round)

        # update global weights
        local_weights_list = copy.deepcopy(local_weights)

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        if round % 10 == 0:
            with torch.no_grad():
                for i in range(args.num_users):
                    print('Test on user {:d}'.format(i))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[i], idxs=user_groups_test[i])
                    local_model_list[i].eval()
                    acc, loss = local_test.test_inference(i, args, backbone_list, local_model_list[i])
                    summary_writer.add_scalar('Test/Acc/user' + str(i), acc, round)

    acc_mtx = torch.zeros([args.num_users])
    loss_mtx = torch.zeros([args.num_users])
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
            local_model_list[idx].eval()
            acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_list[idx])
            acc_mtx[idx] = acc
            loss_mtx[idx] = loss

    print('For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(torch.mean(acc_mtx), torch.std(acc_mtx)))

    return acc_mtx

def FedAvg(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):

    train_loss, train_accuracy = [], []
    global_model = local_model_list[0]

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round} |\n')
        print(datetime.now())
        if args.num_users <= 20:
            idxs_users = np.arange(args.num_users)
        else:
            idxs_users = np.random.choice(args.num_users, 20, replace=False)
        for idx in idxs_users:
            local_node = LocalUpdate(args=args, dataset=train_dataset_list[idx],idxs=user_groups[idx])
            w, loss = local_node.update_weights(idx, backbone_list=backbone_list, model=copy.deepcopy(global_model), global_round=round)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss, round)

        # update global weights
        local_weights_list = average_weights(local_weights)
        global_model = copy.deepcopy(local_model_list[0])
        global_model.load_state_dict(local_weights_list[0], strict=True)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        if round % 5 == 0:
            with torch.no_grad():
                for idx in range(args.num_users):
                    print('Test on user {:d}'.format(idx))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
                    local_model_list[idx] = copy.deepcopy(global_model)
                    local_model_list[idx].eval()
                    acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_list[idx])
                    summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)

    acc_mtx = torch.zeros([args.num_users])
    loss_mtx = torch.zeros([args.num_users])
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
            local_model_list[idx] = copy.deepcopy(global_model)
            local_model_list[idx].eval()
            acc, loss = local_test.test_inference(idx, args, backbone_list, local_model_list[idx])
            acc_mtx[idx] = acc
            loss_mtx[idx] = loss

    print('For all users, mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(torch.mean(acc_mtx), torch.std(acc_mtx)))

    return acc_mtx

def FedPCL(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list):
    global_protos = {}
    global_avg_protos = {}
    local_protos = {}

    for round in tqdm(range(args.rounds)):
        print(f'\n | Global Training Round : {round} |\n')
        local_weights, local_loss1, local_loss2, local_loss_total,  = [], [], [], []
        idxs_users = np.arange(args.num_users)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset_list[idx], idxs=user_groups[idx])
            w, w_urt, loss, protos = local_model.update_weights_lg(args, idx, global_protos, global_avg_protos, backbone_list=backbone_list, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)
            if args.add_noise_proto:
                agg_protos = add_noise_proto(args.device, agg_protos, args.scale, args.perturb_coe, args.noise_type)

            local_weights.append(copy.deepcopy(w))
            local_loss1.append(copy.deepcopy(loss['1']))
            local_loss2.append(copy.deepcopy(loss['2']))
            local_loss_total.append(copy.deepcopy(loss['total']))
            local_protos[idx] = copy.deepcopy(agg_protos)

            summary_writer.add_scalar('Train/Loss/user' + str(idx), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx), loss['2'], round)

        for idx in idxs_users:
            local_model_list[idx].load_state_dict(local_weights[idx])

        # update global protos
        global_avg_protos = proto_aggregation(local_protos)
        global_protos = copy.deepcopy(local_protos)
        loss_avg = sum(local_loss_total) / len(local_loss_total)
        print('| Global Round : {} | Avg Loss: {:.3f}'.format(round, loss_avg))
        summary_writer.add_scalar('Train/Loss/avg', loss_avg, round)

        if round % 20 == 0:
            with torch.no_grad():
                for idx in range(args.num_users):
                    print('Test on user {:d}'.format(idx))
                    local_test = LocalTest(args=args, dataset=test_dataset_list[idx], idxs=user_groups_test[idx])
                    local_model_for_test = copy.deepcopy(local_model_list[idx])
                    local_model_for_test.load_state_dict(local_weights[idx], strict=True)
                    local_model_for_test.eval()
                    acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
                    summary_writer.add_scalar('Test/Acc/user' + str(idx), acc, round)

    acc_mtx = torch.zeros([args.num_users])
    loss_mtx = torch.zeros([args.num_users])
    with torch.no_grad():
        for idx in range(args.num_users):
            print('Test on user {:d}'.format(idx))
            local_test = LocalTest(args = args, dataset = test_dataset_list[idx], idxs = user_groups_test[idx])
            local_model_for_test = copy.deepcopy(local_model_list[idx])
            local_model_for_test.load_state_dict(local_weights[idx], strict=True)
            local_model_for_test.eval()
            acc, loss = local_test.test_inference_twoway(idx, args, global_avg_protos, local_protos[idx], backbone_list, local_model_for_test)
            acc_mtx[idx] = acc
            loss_mtx[idx] = loss

    return acc_mtx

def fed_main(args):
    exp_details(args)

    # set random seed
    args.device = args.device if torch.cuda.is_available() else 'cpu'
    print("Training on", args.device, '...')
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # dataset initialization
    # feature iid, label non-iid
    if args.feature_iid and args.label_iid==0:
        if args.dataset == 'digit':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_mnistm_noniid(args.num_users, args=args)
        elif args.dataset == 'office':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_caltech_noniid(args.num_users, args=args)
        elif args.dataset == 'domainnet':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_real_noniid(args.num_users, args=args)
    # feature non-iid, label iid
    elif args.feature_iid==0 and args.label_iid:
        if args.dataset == 'digit':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_digits(args.num_users, args=args)
        elif args.dataset == 'office':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_office(args.num_users, args=args)
        elif args.dataset == 'domainnet':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_domainnet(args.num_users, args=args)
    # feature non-iid, label non-iid
    elif args.feature_iid==0 and args.label_iid==0:
        if args.dataset == 'digit':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_digits_noniid(args.num_users, args=args)
        elif args.dataset == 'office':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_office_noniid(args.num_users, args=args)
        elif args.dataset == 'domainnet':
            train_dataset_list, test_dataset_list, user_groups, user_groups_test = prepare_data_domainnet_noniid(args.num_users, args=args)

    # load backbone
    if args.model == 'cnn':
        resnet_quickdraw = resnet18(pretrained=True, ds='quickdraw')
        resnet_birds = resnet18(pretrained=True, ds='birds')
        resnet_aircraft = resnet18(pretrained=True, ds='aircraft')
    elif args.model == 'vit':
        vit_t = vit_tiny_patch16_224(pretrained=False)
        vit_t.load_pretrained('./lib/models/weights/Ti_16-i1k-300ep-lr_0.001-aug_light0-wd_0.1-do_0.0-sd_0.0.npz')
        vit_s = vit_small_patch16_224(pretrained=False)
        vit_s.load_pretrained('./lib/models/weights/S_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz')
        vit_b = vit_base_patch16_224(pretrained=False)
        vit_b.load_pretrained('./lib/models/weights/B_16-i1k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.1-sd_0.1.npz')

    # model initialization
    local_model_list = []
    for _ in range(args.num_users):
        if args.num_bb == 1:
            if args.model == 'cnn':
                backbone_list = [resnet_quickdraw]
                local_model = ProjandDeci(512, 256, 10)
            elif args.model == 'vit':
                backbone_list = [vit_s]
                local_model = ProjandDeci(384, 256, 10)
        elif args.num_bb == 3:
            if args.model == 'cnn':
                backbone_list = [resnet_quickdraw, resnet_aircraft, resnet_birds]
                local_model = ProjandDeci(512*3, 256, 10)
            elif args.model == 'vit':
                backbone_list = [vit_t, vit_s, vit_b]
                local_model = ProjandDeci(192+384+768, 256, 10)
            elif args.model == 'other':
                MLP=mlp_m(pretrained=True)
                AlexNet=alexnet(pretrained=True)
                VGG=vgg11(pretrained=True)
                backbone_list = [MLP, AlexNet, VGG]
                local_model = ProjandDeci(4352, 256, 10)
        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    for backbone in backbone_list:
        backbone.to(args.device)
        backbone.eval()

    print(args)
    summary_writer = SummaryWriter('./tensorboard/' + args.dataset + '_' + args.alg + '_' + str(len(backbone_list)) + 'bb_' + str(args.rounds) + 'r_' + str(args.num_users) + 'u_'+ str(args.train_ep) + 'ep')
    if args.alg == 'fedpcl':
        acc_mtx = FedPCL(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list)
    elif args.alg == 'fedavg':
        acc_mtx = FedAvg(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list)
    elif args.alg == 'local':
        acc_mtx = Local(args, summary_writer, train_dataset_list, test_dataset_list, user_groups, user_groups_test, backbone_list, local_model_list)

    return acc_mtx

if __name__ == '__main__':
    num_trial = 3
    args = args_parser()
    acc_mtx = torch.zeros([num_trial, args.num_users])

    for i in range(num_trial):
        args.seed = i
        acc_mtx[i,:] = fed_main(args)

    print("The avg test acc of all trials are:")
    for j in range(args.num_users):
        print('{:.2f}'.format(torch.mean(acc_mtx[:,j])*100))

    print("The stdev of test acc of all trials are:")
    for j in range(args.num_users):
        print('{:.2f}'.format(torch.std(acc_mtx[:,j])*100))

    acc_avg = torch.zeros([num_trial])
    for i in range(num_trial):
        acc_avg[i] = torch.mean(acc_mtx[i,:]) * 100
    print("The avg and stdev test acc of all clients in the trials:")
    print('{:.2f}'.format(torch.mean(acc_avg)))
    print('{:.2f}'.format(torch.std(acc_avg)))
