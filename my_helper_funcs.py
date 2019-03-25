
import numpy as np
from sklearn.metrics import roc_curve, auc

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as line
plt.rcParams.update({'font.size': 22})

def find_correlation(data, threshold=0.6, remove_negative=False):
    """
    """
    corr_mat = data.corr()
    if remove_negative:
        corr_mat = np.abs(corr_mat)
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][corr_mat[col] > threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat

def plot_iv(df, name='iv'):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    x_pos = df[name]
    y_pos = np.arange(len(df))
    ax.barh(y_pos, x_pos, align='center', color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.VAR_NAME)
    ax.invert_yaxis() 
    ax.set_xlabel('Performance')
    ax.set_title('Information Value')
#     plt.plot([2.5, 2.5], [2.5,20], lw=1)
    
    plt.axvspan(0.5, max(df[name]+0.1), color='r', alpha=0.3, lw=0)
    plt.axvspan(0.3, 0.5, color='g', alpha=0.3, lw=0)
    plt.axvspan(0.1, 0.3, color='y', alpha=0.3, lw=0)
    plt.axvspan(0.02,0.1, color='orange', alpha=0.3, lw=0)
    plt.axvspan(0.0,0.02, color='grey', alpha=0.3, lw=0)
    
    plt.show()

def hist_maker(xall, xdef, name, hbins):
    plt.style.use('seaborn-white')
    
    kwargs = dict(#histtype='stepfilled', 
                  alpha=0.3, bins = hbins, #[18, 23, 28, 32, 37, 45, 50, 60, 75, 90],
                  range = (xall.min(), xall.max()), log=True, color='blue')

    fig, ax = plt.subplots(figsize=(20,10))

    count_defs = ax.hist(xdef, **kwargs)
    count_alls = ax.hist(xall, **kwargs)

    ratio_defs = count_defs[0] / count_alls[0]
    ratio_err  = \
        (count_alls[0] * np.sqrt(count_defs[0]) + 
         count_defs[0] * np.sqrt(count_alls[0])) / \
        (count_alls[0]*count_alls[0])

    ratio_xs   = count_defs[1][1:]
    ratio_xs   = ratio_xs - (ratio_xs[1] - ratio_xs[0])/2
    
    ax.set_xlabel(name)
    ax.set_ylabel('# events')
#     ax.gird()
    ax.set_axisbelow(True)
    ax.grid(linestyle='-', linewidth='0.5', color='black')
    ax.axhline(4.2e4, color="gray")
    
    ax2 =ax.twinx()
    ax2.plot(ratio_xs, ratio_defs, 'ro')
    ax2.errorbar(ratio_xs, ratio_defs, yerr=ratio_err, fmt='o', ecolor='orangered',
            color='red', capsize=10, markersize=10)
    ax2.set_ylabel('positive events %')
    
    sns.despine(ax=ax, right=True, left=True)
    sns.despine(ax=ax2, left=True, right=False)
    ax2.spines['right'].set_color('red')
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_roc(roc):
    
    fpr, tpr, _ = roc
    roc_auc = auc(fpr, tpr)
    
    print(roc_auc)
    print(len(fpr))
    print(len(tpr))
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='red') ####,label='ROC curve (area = ') #'%0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()    

def plot_income(y_pred, y_test):
    xs = np.linspace(0,1,100)
    
    print('auc score: ', metrics.roc_auc_score(y_test, y_pred))
    
    imp_model = []
    imp_init  = []
    
    for x in xs:
        yt = (y_pred>x)*1
        conf = metrics.confusion_matrix(y_test,  yt)
        imp_model.append(conf[1][1]*(80-8) - conf[0][1]*8)
        imp_init.append((conf[1][1]+conf[1][0])*(80-8) - (conf[0][0]+conf[0][1])*8)
    
    print(max(imp_model), ' ', max(imp_init))
    
    plt.figure(figsize=(10,8))
    plt.plot(xs, imp_model, 'r--', xs, imp_init, 'b--')
    plt.xlabel('threshold')
    plt.ylabel('economic importance')
    plt.grid(True)
    plt.show()
