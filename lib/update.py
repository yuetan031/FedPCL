#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from losses import ConLoss
from utils import add_noise_img

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if isinstance(image, torch.Tensor):
            image = image.clone().detach()
        else:
            # image = torch.tensor(image)
            image = image
        if isinstance(label, torch.Tensor):
            label = label.clone().detach()
        else:
            label = torch.tensor(label)
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.device
        self.criterion_CE = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        idxs_train = idxs[:int(len(idxs))]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True, drop_last=True)

        return trainloader

    def update_weights(self, idx, backbone_list, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images = images[0]
                images, labels = images.to(self.device), labels.to(self.device)

                # generate representations by different backbone
                for i in range(len(backbone_list)):
                    backbone = backbone_list[i]
                    if i == 0:
                        reps = backbone(images)
                    else:
                        reps = torch.cat((reps, backbone(images)), 1)

                # compute CE loss
                model.zero_grad()
                log_probs, _ = model(reps)
                loss = self.criterion_CE(log_probs, labels)

                # SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                        global_round, idx, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights_lg(self, args, idx, global_protos, global_avg_protos, backbone_list, model, global_round=round):
        # Set mode to train model
        model.train()
        epoch_loss = {'total':[],'1':[], '2':[]}
        loss_mse = nn.MSELoss().to(args.device)
        criterion_CL = ConLoss(temperature=0.07)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.train_ep):
            batch_loss = {'1':[],'2':[],'total':[]}
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if args.add_noise_img:
                    images[0] = add_noise_img(images[0], args.scale, args.perturb_coe, args.noise_type)
                    images[1] = images[0]
                images = torch.cat([images[0], images[1]], dim=0)
                images, labels = images.to(self.device), labels.to(self.device)

                # generate representations by different backbone
                with torch.no_grad():
                    for i in range(len(backbone_list)):
                        backbone = backbone_list[i]
                        if i == 0:
                            reps = backbone(images)
                        else:
                            reps = torch.cat((reps, backbone(images)), 1)

                # compute supervised contrastive loss
                model.zero_grad()
                log_probs, features = model(reps)
                bsz = labels.shape[0]
                lp1, lp2 = torch.split(log_probs, [bsz, bsz], dim=0)
                loss1 = self.criterion_CE(lp1, labels)

                # compute regularized loss term
                loss2 = 0 * loss1
                if len(global_protos) == args.num_users:
                    if args.alg == 'fedproto':
                        # compute global proto-based distance loss
                        num, xdim = features.shape
                        features_global = torch.zeros_like(features)
                        for i, label in enumerate(labels):
                            features_global[i, :] = copy.deepcopy(global_protos[label.item()].data)
                        loss2 = loss_mse(features_global, features) / num * args.ld
                    elif args.alg == 'fedpcl':
                        # compute global proto based CL loss
                        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                        for i in range(args.num_users):
                            for label in global_avg_protos.keys():
                                if label not in global_protos[i].keys():
                                    global_protos[i][label] = global_avg_protos[label]
                            loss2 += criterion_CL(features, labels, global_protos[i])

                loss = loss2

                # SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss['1'].append(loss1.item())
                batch_loss['2'].append(loss2.item())
                batch_loss['total'].append(loss.item())
                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | User: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tLoss2: {:.3f}'.format(
                            global_round, idx, iter, batch_idx * len(images),
                                    len(self.trainloader.dataset),
                                    100. * batch_idx / len(self.trainloader),
                                    loss.item(),
                                    loss2.item()))

            epoch_loss['1'].append(sum(batch_loss['1']) / len(batch_loss['1']))
            epoch_loss['2'].append(sum(batch_loss['2']) / len(batch_loss['2']))
            epoch_loss['total'].append(sum(batch_loss['total']) / len(batch_loss['total']))

        epoch_loss['1'] = sum(epoch_loss['1']) / len(epoch_loss['1'])
        epoch_loss['2'] = sum(epoch_loss['2']) / len(epoch_loss['2'])
        epoch_loss['total'] = sum(epoch_loss['total']) / len(epoch_loss['total'])

        # generate representation
        agg_protos_label = {}
        model.eval()
        for batch_idx, (images, label_g) in enumerate(self.trainloader):
            images = images[0]
            images, labels = images.to(self.device), label_g.to(self.device)

            with torch.no_grad():
                for i in range(len(backbone_list)):
                    backbone = backbone_list[i]
                    if i == 0:
                        reps = backbone(images)
                    else:
                        reps = torch.cat((reps, backbone(images)), 1)
            _, features = model(reps)
            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(features[i, :])
                else:
                    agg_protos_label[labels[i].item()] = [features[i, :]]

        return model.state_dict(), [], epoch_loss, agg_protos_label

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion_SupCL(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

    def generate_protos(self, backbone_list, model):
        model.eval()
        agg_protos_label = {}
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images = images[0]
            images, labels = images.to(self.device), labels.to(self.device)

            for i in range(len(backbone_list)):
                backbone = backbone_list[i]
                if i == 0:
                    reps = backbone(images)
                else:
                    reps = torch.cat((reps, backbone(images)), 1)
            _, features = model(reps)
            for i in range(len(labels)):
                if labels[i].item() in agg_protos_label:
                    agg_protos_label[labels[i].item()].append(features[i, :])
                else:
                    agg_protos_label[labels[i].item()] = [features[i, :]]

        return agg_protos_label

class LocalTest(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.testloader = self.test_split(args, dataset, list(idxs))
        self.device = args.device
        self.criterion = nn.NLLLoss().to(args.device)

    def test_split(self, args, dataset, idxs):
        idxs_test = idxs[:int(1 * len(idxs))]

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                 batch_size=args.test_bs, shuffle=False)
        return testloader

    def test_inference(self, idx, args, backbone_list, local_model):
        device = args.device
        criterion = nn.NLLLoss().to(device)

        model = local_model
        model.to(args.device)
        loss, total, correct = 0.0, 0.0, 0.0

        # test (only use local model)
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(device), labels.to(device)

            # generate representations by different backbone
            for i in range(len(backbone_list)):
                backbone = backbone_list[i]
                if i == 0:
                    reps = backbone(images)
                else:
                    reps = torch.cat((reps, backbone(images)), 1)

            probs, _ = model(reps)

            # prediction
            _, pred_labels = torch.max(probs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        loss /= (batch_idx + 1)
        print('| User: {} | Test Acc: {:.5f} | Test Loss: {:.5f}'.format(idx, acc, loss))

        return acc, loss

    def test_inference_twoway(self, idx, args, global_protos, local_protos, backbone_list, local_model):
        device = args.device
        criterion = nn.NLLLoss().to(device)
        loss_mse = nn.MSELoss()

        model = local_model
        model.to(args.device)
        loss, total, correct = 0.0, 0.0, 0.0

        # test (only use local model)
        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(device), labels.to(device)

            # generate representations by different backbone
            for i in range(len(backbone_list)):
                backbone = backbone_list[i]
                if i == 0:
                    reps = backbone(images)
                else:
                    reps = torch.cat((reps, backbone(images)), 1)

            probs, features = model(reps)

            # compute the dist between features and input protos
            a_large_num = 100
            dist = a_large_num * torch.ones(size=(images.shape[0], args.num_classes)).to(device)  # initialize a distance matrix
            for i in range(images.shape[0]):
                for j in range(args.num_classes):
                    if j in local_protos.keys():
                        d = loss_mse(features[i, :], local_protos[j]) # compare with local protos
                        dist[i, j] = d
            _, pred_labels = torch.min(dist, 1)

            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        acc = correct / total
        loss /= (batch_idx + 1)
        print('| User: {} | Test Acc: {:.5f} | Test Loss: {:.5f}'.format(idx, acc, loss))

        return acc, loss

def save_protos(round, args, backbone_list, local_model_list, train_dataset_list, user_groups, global_protos):
    """ Returns the test accuracy and loss.
    """
    device = args.device

    agg_protos_label = {}
    for idx in range(1):
        agg_protos_label[idx] = {}
        model = local_model_list[idx]
        model.eval()
        model.to(args.device)
        trainloader = DataLoader(DatasetSplit(train_dataset_list[idx], user_groups[idx]), batch_size=32, shuffle=True, drop_last=True)
        for batch_idx, (images, labels) in enumerate(trainloader):
            images = images[0]
            images, labels = images.to(device), labels.to(device)

            # generate representations by different backbone
            for i in range(len(backbone_list)):
                backbone = backbone_list[i]
                if i == 0:
                    reps = backbone(images)
                else:
                    reps = torch.cat((reps, backbone(images)), 1)

            # compute features
            model.zero_grad()
            _, protos = model(reps)

            for k in range(len(labels)):
                if labels[k].item() in agg_protos_label[idx].keys():
                    agg_protos_label[idx][labels[k].item()].append(protos[k, :])
                else:
                    agg_protos_label[idx][labels[k].item()] = [protos[k, :]]

    x = []
    y = []
    u = []
    for idx in range(1):
        for label in [0]:
            for proto in agg_protos_label[idx][label]:
                if args.device == 'cuda':
                    tmp = proto.cpu().detach().numpy()
                else:
                    tmp = proto.detach().numpy()
                x.append(tmp)
                y.append(label)
                u.append(round)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    np.save('./save/local_protos_' + str(round) + 'r_' + args.alg + '_protos.npy', x)
    np.save('./save/local_protos_' + str(round) + 'r_' + args.alg + '_labels.npy', y)
    np.save('./save/local_protos_' + str(round) + 'r_' + args.alg + '_rounds.npy', u)

    xx = []
    yy = []
    uu = []
    for label in [0]:
        if args.device == 'cuda':
            xx.append(global_protos[label].cpu().detach().numpy())
        else:
            xx.append(global_protos[label].detach().numpy())
        yy.append(label)
        uu.append(round)
    np.save('./save/global_protos_' + str(round) + 'r_' + args.alg + '_protos.npy', xx)
    np.save('./save/global_protos_' + str(round) + 'r_' + args.alg + '_labels.npy', yy)
    np.save('./save/global_protos_' + str(round) + 'r_' + args.alg + '_rounds.npy', uu)

    print("Save protos and labels successfully.")
