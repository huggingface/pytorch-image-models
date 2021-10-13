""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from sklearn.metrics import precision_score,accuracy_score ,recall_score ,log_loss,f1_score,confusion_matrix
import torch.nn.functional as F
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
def precision(output, target):
    _, y_pred = output.topk(1, 1, True, True)
    y_true = target
    y_true=y_true.tolist()
    y_pred=y_pred.tolist()
    y_pred=sum(y_pred, [])
    TP=0+0.00000000000000009
    FP=0
    for i in range(len(y_true)):
        if y_true[i]==1 and y_pred[i]==1:
            TP=TP+1
        if y_true[i]==0 and y_pred[i]==1:
            FP=FP+1
    precision=precision_score(y_true,y_pred)

    return precision*100


def recall(output, target):
    y_pred = torch.ge(output, 0.0)
    _, y_pred = output.topk(1, 1, True, True)
    y_true = target
    true_positive =  len((y_true.flatten() == y_pred.flatten()).nonzero().flatten())
    y_true=y_true.tolist()
    y_pred=y_pred.tolist()
    y_pred=sum(y_pred, [])
    TP=0+0.00000000000000009
    FN=0
    for i in range(len(y_true)):
        if y_true[i]==1 and y_pred[i]==1:
            TP=TP+1
        if y_true[i]==1 and y_pred[i]==0:
            FN=FN+1
    recall =recall_score(y_true,y_pred)
    return recall*100

def f1_scor(output, target):
    y_pred = torch.ge(output, 0.5)
    _, y_pred = output.topk(1, 1, True, True)
    y_true = target
    y_true=y_true.tolist()
    y_pred=y_pred.tolist()
    return f1_score(y_true,y_pred)*100
