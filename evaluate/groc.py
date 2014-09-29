import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import as_roc
import os

#def GetColor():
#    colors=('bo-', 'g^-', 'm*-', 'r>-', 'c<-')
#    return colors

def GetColor():
    colors=('b-', 'g-', 'm-', 'r-', 'c-')
    return colors

def GetName():
    name = []
    name.append('RLFM')
    name.append('AS')
    name.append('TBM')
    name.append('CRBM')
    return name

def GetFigDesDir():
    return '/home/lijk/working/project/localrepos/paper/preprint/coldstart/figure/'

def PlotData(real, predict, color, label):
    real = np.array(real) 
    predict = np.array(predict)
    fpr, tpr, thresholds = metrics.roc_curve(real, predict, pos_label = 2)
    auc = metrics.auc(fpr, tpr)
    print "AUC= ", auc
    s = str(float(int(auc * 1000)) / 1000)
    plt.plot(fpr, tpr, color, label = label + ' (AUC ' + s + ')')

def LoadGROC(path):
    fr = open(path)
    f = []
    for l in fr:
        f.append(l.strip().split(' '))
    fr.close()
    for i in range(len(f[0])):
        f[0][i] = float(f[0][i])
        f[1][i] = float(f[1][i])
    return f[0], f[1]

def LoadRLF(path):
    f = open(path)
    real = []
    pre = []
    for l in f:
        t = l.split('\t')
        real.append(float(t[0].strip()) + 1)
        pre.append(float(t[1].strip()))
    f.close()
    return real, pre

def main():
    name = 'tbm_task1'
    RatingGROCMain(name)
    name = 'tbm_task2'
    RatingGROCMain(name)
    name = 'tbm_task3'
    RatingGROCMain(name)

def RatingGROCPlot(name, color, real, pre, path):
    plt.clf()
    for i in range(len(name)):
        PlotData(real[i], pre[i], color[i], name[i])
    plt.legend(loc = 'lower right')
    plt.ylabel("True Positive Rate")
    #plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.axis([0,1,0,1])
    plt.savefig(path)
    plt.clf()

def CRBMTask3():
    rlf_path = 'data/roc/rlf/task3_groc'
    real = []
    pre = []
    r, p = LoadRLF(rlf_path)
    real.append(r)
    pre.append(p)
 
    r, p = as_roc.AsTask3GROC()
    real.append(r)
    pre.append(p)

    path = 'data/roc/tbm_task3_groc'
    r, p = LoadGROC(path)
    real.append(r)
    pre.append(p)

    path = 'data/roc/crbm/crbm_task3_groc'
    r, p = LoadGROC(path)
    real.append(r)
    pre.append(p)
    
    fig_path = os.path.join(GetFigDesDir(), 'task3_groc.eps')
    RatingGROCPlot(GetName(), GetColor(), real, pre, fig_path)

def CRBMTask2():
    rlf_path = 'data/roc/rlf/task2_groc'
    real = []
    pre = []
    r, p = LoadRLF(rlf_path)
    real.append(r)
    pre.append(p)
 
    r, p = as_roc.AsTask2GROC()
    real.append(r)
    pre.append(p)

    path = 'data/roc/tbm_task2_groc'
    r, p = LoadGROC(path)
    real.append(r)
    pre.append(p)

    path = 'data/roc/crbm/crbm_task2_groc'
    r, p = LoadGROC(path)
    real.append(r)
    pre.append(p)
    
    fig_path = os.path.join(GetFigDesDir(), 'task2_groc.eps')
    RatingGROCPlot(GetName(), GetColor(), real, pre, fig_path)

def CRBMTask1():
    rlf_path = 'data/roc/rlf/task1_groc'
    real = []
    pre = []
    r, p = LoadRLF(rlf_path)
    real.append(r)
    pre.append(p)
 
    r, p = as_roc.AsTask1GROC()
    real.append(r)
    pre.append(p)

    path = 'data/roc/tbm_task1_groc'
    r, p = LoadGROC(path)
    real.append(r)
    pre.append(p)

    path = 'data/roc/crbm/crbm_task1_groc'
    r, p = LoadGROC(path)
    real.append(r)
    pre.append(p)
    
    fig_path = os.path.join(GetFigDesDir(), 'task1_groc.eps')
    RatingGROCPlot(GetName(), GetColor(), real, pre, fig_path)

#main()
#CRBM()
CRBMTask1()
CRBMTask2()
CRBMTask3()
#PlotRLF()
