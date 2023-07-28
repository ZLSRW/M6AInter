import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#将预测概率转换为预测标签。
def Prediction_label(pred):
    a,b,c=pred.size()
    pred_arry=pred.detach().numpy()
    # print(str(a)+' '+str(b)+' '+str(c))
    predlabels=np.zeros((a,b,c))
    i=0
    while i<a:
        j=0
        while j<b:
            k=0
            while k<c:
                if pred_arry[i][j][k]>0.5:
                    predlabels[i][j][k]=1.0
                else:
                    predlabels[i][j][k]=0.0
                k+=1
            j+=1
        i+=1
    return torch.from_numpy(predlabels)

#attention mask
def Zero_One_Mask(pred):
    a,b,c=pred.size()
    pred_arry=pred.detach().numpy()
    # print(str(a)+' '+str(b)+' '+str(c))
    i=0
    while i<a:
        j=0
        while j<b:
            k=0
            while k<c:
                if pred_arry[i][j][k]>0.5:
                    pred_arry[i][j][k]=1.0
                else:
                    pred_arry[i][j][k]=0.0
                k+=1
            j+=1
        i+=1
    return torch.from_numpy(pred_arry)

#index mask
def Index_Mask(pred,inputs_site,scale=0.5):
    a, b, c = pred.size()
    # print(type(scale))
    d=math.ceil(c*scale)

    a1,b1,c1 = inputs_site.size()
    pred_arry = pred.detach().numpy()
    inputs_site = inputs_site.detach().numpy()
    # 获得mask后的训练概率和训练标签
    pred_mask = np.zeros((a, b, d))
    pred_mask_index = np.zeros((a, b, d))
    inputs_site_mask = np.zeros((a1, b1, d))

    i=0
    while i < a:
        j = 0
        while j < b:
            k = 0
            counter=0
            index=0
            while k < c:
                #根据scale对预测序列和标签序列进行裁切
                if inputs_site[i][j][k] == 1.0 and counter<=3:
                    counter+=1
                    pred_mask_index[i][j][index] = k #记录正样本位置
                    pred_mask[i][j][index]=pred_arry[i][j][k]
                    inputs_site_mask[i][j][index]=inputs_site[i][j][k]
                    index+=1
                k += 1
            # print(str(inputs_site[i][j])+' '+str(counter))
            while index < d:
                n=0
                while n < len(inputs_site[i][j]):
                    if inputs_site[i][j][n]==0.0:
                        pred_mask_index[i][j][index] = n  # 记录正样本位置
                        pred_mask[i][j][index] = pred_arry[i][j][n]
                        inputs_site_mask[i][j][index] = inputs_site[i][j][n]
                        index += 1
                        break
                    n+=1



            j += 1
        i += 1

    return torch.from_numpy(pred_mask_index).float(),torch.from_numpy(pred_mask).float(),torch.from_numpy(inputs_site_mask) .float()#下标，概率，位点与否

def Sequence_reduction(pred,inputs_site,ID_len): #将窗口中的转态全部转换为二维列表，左边为状态，右边为标签
    a, b, c = pred.size()
    pred_arry = pred.detach().numpy()
    inputs_site = inputs_site.detach().numpy()
    ID_len_array=ID_len.detach().numpy()
    # 获得mask后的训练概率和训练标签

    #用于指标计算
    prediction=[]
    label=[]
    #用于序列还原
    ID_prediction=[]
    ID_label=[]

    i=0
    while i < a:
        j = 0
        while j < b:
            k = 0

            while k < int(ID_len_array[i][j][1]):
                temp_id_pre = []
                temp_id_lab = []
                #根据scale对预测序列和标签序列进行裁切
                prediction.append(float(pred_arry[i][j][k]))
                label.append(float(inputs_site[i][j][k]))

                temp_id_pre.append(ID_len_array[i][j][0])
                temp_id_pre.append(float(pred_arry[i][j][k]))

                temp_id_lab.append(ID_len_array[i][j][0])
                temp_id_lab.append(float(inputs_site[i][j][k]))

                ID_prediction.append(temp_id_pre)
                ID_label.append(temp_id_lab)

                k += 1
            j += 1
        i += 1

    return prediction,label,ID_prediction,ID_label

def Sequence_reduction1(pred,inputs_site,ID_len): #将窗口中的转态全部转换为二维列表，左边为状态，右边为标签
    a, b, c = pred.size()
    pred_arry = pred.detach().numpy()
    inputs_site = inputs_site.detach().numpy()
    ID_len_array=ID_len.detach().numpy()
    # 获得mask后的训练概率和训练标签

    #用于指标计算
    prediction=[]
    label=[]
    #用于序列还原
    ID_prediction=[]
    ID_label=[]

    i=0
    while i < a:
        j = 0
        while j < b:
            k = 0
            while k < c:
                temp_id_pre = []
                temp_id_lab = []
                #根据scale对预测序列和标签序列进行裁切
                prediction.append(float(pred_arry[i][j][k]))
                label.append(float(inputs_site[i][j][k]))

                temp_id_pre.append(ID_len_array[i][j][0])
                temp_id_pre.append(float(pred_arry[i][j][k]))

                temp_id_lab.append(ID_len_array[i][j][0])
                temp_id_lab.append(float(inputs_site[i][j][k]))

                ID_prediction.append(temp_id_pre)
                ID_label.append(temp_id_lab)

                k += 1
            j += 1
        i += 1

    return prediction,label,ID_prediction,ID_label




#多维张量指定维度对角化
def tensor_diag(input):
    a,b=input.size()
    input_arry=input.detach().numpy()
    input_diag_array=np.zeros((a,b,b))
    # print(str(a)+' '+str(b)+' '+str(c))
    i=0
    while i<a:
        j=0
        while j<b:
            input_diag_array[i][j][j]=input_arry[i][j]
            j+=1
        i+=1
    return torch.from_numpy(input_diag_array).float()

#最大最小归一化以及硬标签转换
def max_min(input):
    a, b, c = input.size()
    threahold=0.6
    input_array = input.detach().numpy()
    i = 0
    while i < a:
        j = 0
        while j < b:
            k=0
            max_x=max(input_array[i][j])
            min_x=min(input_array[i][j])
            # print('最大值 '+str(max_x)+' 最小值 '+str(min_x))
            diff=max_x-min_x
            while k<c:
                input_array[i][j][k]=((input_array[i][j][k])-min_x)/diff
                if input_array[i][j][k]<threahold:
                    input_array[i][j][k]=0.0
                k+=1
            j += 1

        i += 1
    return torch.from_numpy(input_array).float()


def max_min_2D(input):
    a, b = input.size()
    threahold=0.5
    input_array = input.detach().numpy()
    i = 0
    while i < a:
        j = 0
        max_x=max(input_array[i])
        min_x=min(input_array[i])
        diff=max_x-min_x
        while j<b:
            input_array[i][j]=((input_array[i][j])-min_x)/diff
            if input_array[i][j]<threahold:
                input_array[i][j]=0.0

            j += 1
        i += 1
    return torch.from_numpy(input_array).float()


def pos_neg_mask(input,pos_mask,neg_mask):
    a, b = input.size()
    input_array = input.detach().numpy()
    pos_mask = pos_mask.detach().numpy()
    neg_mask = neg_mask.detach().numpy()
    i = 0
    while i < a:
        j = 0
        while j < b:
            if input_array[i][j]<0:
                neg_mask[i][j]=0
            else:
                pos_mask[i][j]=0
            j += 1
        i += 1
    return torch.from_numpy(pos_mask).float(),torch.from_numpy(neg_mask).float()

def find_duplicates(lst):
    indices = {}
    for i, x in enumerate(lst):
        if x not in indices:
            indices[x] = [i]
        else:
            indices[x].append(i)
    return [v for v in indices.values() if len(v) > 1]

def normalize_motif(ids, tensor):

    for row in ids:
        idxs = row
        slices = [tensor[:, idx, :] for idx in idxs]
        mean_ = torch.stack(slices, axis=0).mean()
        for slice_ in slices:
            if mean_ != 0:
                slice_ /= mean_
            else:
                slice_.fill_(0)

    return tensor


def auroc(prob, label):
    y_true = label.data.numpy().flatten()
    y_scores = prob.data.numpy().flatten()
    y_scores=np.nan_to_num(y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auroc_score = auc(fpr, tpr)
    return auroc_score, fpr, tpr


def AUC(prob, label):

    fpr, tpr, thresholds = roc_curve(label, prob)
    auroc_score = auc(fpr, tpr)
    return auroc_score

def auprc(prob,label):
    y_true=label.data.numpy().flatten()
    y_scores=prob.data.numpy().flatten()
    y_scores = np.nan_to_num(y_scores)
    precision,recall,thresholds=precision_recall_curve(y_true,y_scores)
    auprc_score=auc(recall,precision)
    return auprc_score,precision,recall

def AUPR(prob,label):
    precision,recall,thresholds=precision_recall_curve(prob,label)
    auprc_score=auc(recall,precision)
    return auprc_score

def MyRealAndPrediction(Real,prediction):
    RealAndPrediction = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPrediction.append(pair)
        counter = counter + 1
    return RealAndPrediction
def MyRealAndPredictionProb(Real,prediction):
    RealAndPredictionProb = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPredictionProb.append(pair)
        counter = counter + 1
    return RealAndPredictionProb

def Indicator(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    y_real=y_real.data.numpy().flatten()
    y_predict=y_predict.data.numpy().flatten()
    x=[]
    y=[]
    z=[]
    for ele in y_real:
        ele=int(ele)
        x.append(ele)

    for ele in y_predict:
        ele=float(ele)
        z.append(ele)
        if ele >0.5:
            y.append(1)
        else:
            y.append(0)

    RealAndPrediction = MyRealAndPrediction(x, y)
    RealAndPredictionProb = MyRealAndPredictionProb(x, z)

    CM = confusion_matrix(x, y)
    print(CM)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]

    Acc = (TN + TP) / (TN + TP + FN + FP)
    if (TP+FN)!=0:
        Sen = TP / (TP + FN)
    else:
        Sen=np.inf
    if (TN + FP)!=0:
        Spec = TN / (TN + FP)
    else:
        Spec=np.inf
    if (TP + FP)!=0:
        Prec = TP / (TP + FP)
    else:
        Prec=np.inf
    if math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))!=0:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC=np.inf
    f1 = f1_score(y, x)
    Auc,_,_= auroc(torch.tensor(y_predict),torch.tensor(x))
    Aupr,_,_=auprc(torch.tensor(y_predict),torch.tensor(x))

    Result = []
    Result.append(round(MCC, 4))
    Result.append(round(Auc, 4))
    Result.append(round(Aupr, 4))
    Result.append(round(f1, 4))
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))

    return Result,RealAndPrediction,RealAndPredictionProb

def MyConfusionMatrix(y_real, y_predict):
    from sklearn.metrics import confusion_matrix
    y_real = y_real
    y_predict = y_predict

    x = []
    y = []
    z = []
    for ele in y_real:
        ele = int(ele)
        x.append(ele)

    for ele in y_predict:
        ele = float(ele)
        z.append(ele)
        if ele > 0.5:
            y.append(1)
        else:
            y.append(0)

    CM = confusion_matrix(x, y)
    print(CM)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    if (TP + FN) != 0:
        Sen = TP / (TP + FN)
    else:
        Sen = np.inf
    if (TN + FP) != 0:
        Spec = TN / (TN + FP)
    else:
        Spec = np.inf
    if (TP + FP) != 0:
        Prec = TP / (TP + FP)
    else:
        Prec = np.inf
    if math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) != 0:
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        MCC = np.inf
    f1 = f1_score(y, x)
    Auc = AUC(torch.tensor(y_predict), torch.tensor(x))
    Aupr = AUPR(torch.tensor(y), torch.tensor(x))

    print('MCC:'+str(round(MCC, 4))+' AUC:'+str(round(Auc, 4))+' AUPR:',str(round(Aupr, 4))+
          ' F1:'+str(round(f1, 4))+' Acc:'+str(round(Acc, 4)) + ' Sen:'+ str(round(Sen, 4))+' Spec:', str(round(Spec, 4))+' Prec:'+str(round(Prec, 4)))

    Result = [] #len=8
    Result.append(round(MCC, 4))
    Result.append(round(Auc, 4))
    Result.append(round(Aupr, 4))
    Result.append(round(f1, 4))
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    return Result

#标准化
def normalized_input(input):
    a,b,c=input.size()
    input=input.detach().numpy()
    # print(str(a)+' '+str(b)+' '+str(c))
    new_input=np.zeros((a,b,c))
    i=0
    while i<a:
        j=0
        while j<b:
            k=0
            while k<c:
                new_input[i][j][k]=(input[i][j][k]-min(input[i][j]))/(max(input[i][j])-min(input[i][j]))
                k+=1
            j+=1
        i+=1
    return torch.from_numpy(new_input).float()