
import warnings
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss



class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()
        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]
            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)
        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C
            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)
        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss

class DefaultConfig(object):
    env = '552552' # visdom 环境
    vis_port =8097
    model = 'unet_3d' # 使用的模型，名字�
须与models/__init__.py中的名字一�?    
    train_data_root = r"/media/hitlab/GuoXuTao/FCN01/MICCAI_BraTS17_Data_Training" # 训练集存放路�?    test_data_root = r'/media/hitlab/GuoXuTao/FCN01/MICCAI_BraTS17_Data_Training' # 测试集存放路�?    load_model_path = False # � 载预训练的模型的路径，为None代表不� �?
    batch_size = 4 # batch size
    use_gpu = True # user GPU or not
    num_workers = 8 # how many workers for loading data
    print_freq = 20 # print info every N batch

    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
      
    max_epoch = 1000000
    lr = 0.0001 # initial learning rate
    lr_decay = 1 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-8 # 损失函数

class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.reset()       # __init__():reset parameters

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

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist

def calc_dice(label_true,label_pred):
    B_index=label_true.shape[0]*label_true.shape[1]*label_true.shape[2]
    A_index=label_pred.shape[0]*label_pred.shape[1]*label_pred.shape[2]
    count=0.0
    for i in range(label_true.shape[0]):
        for j in range(label_true.shape[1]):
            for k in range(label_true.shape[2]):
                if label_true[i][j][k]==label_pred[i][j][k]:
                    count=count+1
    #print count,B_index,A_index
    return float(2*count/(B_index+A_index)),sum(sum(label_true!=label_pred))

def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,}, cls_iu,acc

def cross_entropy2d(input, target, weight=None, size_average=False):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1,c)
    print(log_p)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1,c)
    print(log_p)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def parse(self,kwargs):
        '''
        � �据字�
�kwargs 更新 config参数
        '''
        for k,v in kwargs.iteritems():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print(k,getattr(self,k))


DefaultConfig.parse = parse
opt =DefaultConfig()
# opt.parse = parse
