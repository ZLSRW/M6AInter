import json
import warnings

from data_loader.SiteBinding_dataloader1 import ForecastDataset
from .seq_graphing import Model

import torch.utils.data as torch_data
import time
import os
import csv
import torch.nn as nn
import torch.nn.functional as F
from .Utils import *

warnings.filterwarnings("ignore")

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def save_model(model, model_dir, epoch, fold):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, str(fold)+'_'+epoch + '_PepBindA.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)

def save_model1(model, model_dir, epoch, fold):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, str(fold)+'_'+'best_'+epoch + '_PepBindA.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)

def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_PepBindA.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model

def validate_inference_binding_site(model, dataloader):
    model.eval()
    with torch.no_grad():
        for i, ( inputs, inputs_kmer, inputs_Onehot,inputs_labels) in enumerate(
            dataloader):
            inputs = inputs
            inputs_kmer=inputs_kmer
            inputs_Onehot=inputs_Onehot
            inputs_labels = inputs_labels

            forecast_features,forecast_result,_,motifs,x_type= model(inputs,inputs,inputs_kmer,inputs_Onehot)
            result,Real_Prediction,Real_Prediction_Prob=Indicator(inputs_labels,forecast_result)

            validate_auc, _, _ = auroc(forecast_result, inputs_labels)
            validate_aupr, _, _ = auprc(forecast_result, inputs_labels)
            result[2]=round(validate_aupr,4)

            labels_real = list(inputs_labels.contiguous().view(-1).detach().numpy())
            forecast_features = list(forecast_features.detach().numpy()) #全局特征
            xx = 0
            while xx < len(forecast_features):
                forecast_features[xx]=list(forecast_features[xx])
                forecast_features[xx].append(int(labels_real[xx]))
                xx += 1

    return  result,forecast_features,Real_Prediction,Real_Prediction_Prob,motifs,x_type

def train(train_data, valid_data, args, result_file, fold):
    node_cnt = args.seq_len
    bat=args.batch_size
    print('node_cnt '+str(node_cnt))
    model = Model(bat,args.cluster_num, node_cnt, args.multi_layer)
    model.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')

    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    train_set = ForecastDataset(train_data,seq_size=args.seq_len,seq_size1=args.seq_len1)
    valid_set = ForecastDataset(valid_data,seq_size=args.seq_len,seq_size1=args.seq_len1)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                         num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = torch.nn.BCELoss( reduction='mean')

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    best_validate_Acc= 0.0
    best_result=[]

    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        auc_total=0
        aupr_total=0

        Temp_train_feature=[]

        for i, (
        inputs, inputs_Kmer, inputs_Onehot, inputs_labels) in enumerate(
                train_loader):
            inputs = inputs
            inputs_Kmer=inputs_Kmer
            inputs_Onehot=inputs_Onehot
            inputs_labels = inputs_labels

            forecast_feature,forecast_prob,CLoss,_,_= model(inputs,inputs,inputs_Kmer,inputs_Onehot) #32x12x100 结合位点

            labels_real = list(inputs_labels.contiguous().view(-1).detach().numpy())
            forecast_feature = list(forecast_feature.detach().numpy())
            xx = 0
            while xx < len(forecast_feature):
                forecast_feature[xx]=list(forecast_feature[xx])
                forecast_feature[xx].append(int(labels_real[xx]))
                xx += 1
            Temp_train_feature.extend(forecast_feature)

            train_auc,_,_=auroc(forecast_prob,inputs_labels)
            train_aupr,_,_=auprc(forecast_prob,inputs_labels)

            binding_loss = criterion(forecast_prob, inputs_labels.float())
            all_loss=binding_loss+CLoss
            # all_loss=binding_loss

            auc_total+=train_auc
            aupr_total+=train_aupr

            print('epoch %d,binding_loss %.4f, Gloss %.4f, train_auc %.4f, train_aupr %.4f  '
                  % (epoch + 1, binding_loss, CLoss, train_auc, train_aupr))
            cnt += 1

            # loss.backward()
            model.zero_grad()

            all_loss.backward()

            my_optim.step()

            loss_total += float(all_loss)

        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | train_auc {:5.4f}| train_aupr {:5.4f}'.format(epoch+1, (
                time.time() - epoch_start_time), loss_total / cnt,auc_total/cnt,aupr_total/cnt))

        if (epoch+1)%10==0:
            save_model(model, result_file, epoch, fold)

        if 1==1:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')

            result,validate_features,Real_prediction,Real_prediction_prob,motifs,x_type=validate_inference_binding_site(model, valid_loader)

            motifs = motifs.detach().numpy()
            new_motifs = []
            new_motifs.extend([mo] for mo in motifs)
            x_type = x_type.detach().numpy()

            MCC = result[0]
            auc = result[1]
            aupr=result[2]
            F1 = result[3]
            Acc = result[4]
            Sen = result[5]
            Spec = result[6]
            Prec = result[7]

            print('validate_MCC: '+str(round(MCC,4))+' '+' validate_auc: '+str(round(auc,4))+' validate_aupr: '+str(round(aupr,4))+' '+' validate_F1: '+str(round(F1,4))+' '+
                  ' validate_Acc: '+str(round(Acc,4))+' '+' validate_Sen: '+str(round(Sen,4))+' '+' validate_Spec: '+str(round(Spec,4))+' '
                   +' validate_Prec: '+str(round(Prec,4)))

            if Acc >= best_validate_Acc:
                best_validate_Acc = Acc
                best_result=result

                is_best_for_now = True
            # save model
            if is_best_for_now:
                save_model1(model, result_file,epoch, fold)

    print(
        'best_MCC: ' + str(round(best_result[0], 4)) + ' ' + ' best_auc: ' + str(round(best_result[1], 4)) + ' best_aupr: ' + str(
            round(best_result[2], 4)) + ' ' + ' best_F1: ' + str(round(best_result[3], 4)) + ' ' +
        ' best_Acc: ' + str(round(best_result[4], 4)) + ' ' + ' best_Sen: ' + str(
            round(best_result[5], 4)) + ' ' + ' best_Spec: ' + str(round(best_result[6], 4)) + ' '
        + ' best_Prec: ' + str(round(best_result[7], 4)))
    return forecast_feature,best_result


def test(test_data, args, result_train_file, result_test_file): #
    with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)
    model = load_model(result_train_file)
    test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)
    result,Real_Prediction,Real_Prediction_Prob=validate_inference_binding_site(model, test_loader,)
    MCC = result[0]
    auc = result[1]
    aupr = result[2]
    F1 = result[3]
    Acc = result[4]
    Sen = result[5]
    Spec = result[6]
    Prec = result[7]
    print(
        'validate_MCC: ' + str(round(MCC, 4)) + ' ' + ' validate_auc: ' + str(round(auc, 4)) + ' validate_aupr: ' + str(
            round(aupr, 4)) + ' ' + ' validate_F1: ' + str(round(F1, 4)) + ' ' +
        ' validate_Acc: ' + str(round(Acc, 4)) + ' ' + ' validate_Sen: ' + str(
            round(Sen, 4)) + ' ' + ' validate_Spec: ' + str(round(Spec, 4)) + ' '
        + ' validate_Prec: ' + str(round(Prec, 4)))

