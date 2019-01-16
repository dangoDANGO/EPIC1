
# coding: utf-8

# # Imported Modules

# In[1]:


import os
import sys

import pandas as pd
import numpy as np
import seaborn as sns

from pandas import DataFrame as df
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from matplotlib import pyplot as plt


# # All functions

# In[2]:


def run_tSNE(try_id, data):
    X_embedded = TSNE(
        n_components=2,
        init='pca').fit_transform(data)

    with open('corr_68sig_linc/tSNE_CR_score/tSNE_embedding_' + try_id + '.csv', 'w') as f:
        f.write(',tSNE-1,tSNE-2')
        f.write('\n')
        for i in range(len(X_embedded)):
            f.write(data.index[i] + ',' + str(X_embedded[i][0]) + ',' + str(X_embedded[i][1]) + '\n')

    return


def visual_tSNE(try_id, data, label):
    coordinates = pd.read_csv(
        'corr_68sig_linc/tSNE_CR_score/tSNE_embedding_' + try_id + '.csv',
        header=0,
        index_col=0,
        sep=',')

    coordinates['tSNE-1'] = (coordinates['tSNE-1'] - coordinates['tSNE-1'].min()) / (coordinates['tSNE-1'].max() - coordinates['tSNE-1'].min())
    coordinates['tSNE-2'] = (coordinates['tSNE-2'] - coordinates['tSNE-2'].min()) / (coordinates['tSNE-2'].max() - coordinates['tSNE-2'].min())

    plt.subplots(figsize=(8, 8))
    if label is None:
        plt.scatter(
            coordinates['tSNE-1'], coordinates['tSNE-2'],
            s=20, c='grey', linewidths=0)
    else:
        plt.scatter(
            coordinates['tSNE-1'], coordinates['tSNE-2'],
            s=20, c=data[label], linewidths=0,
            vmin=-1, vmax=1, cmap=plt.cm.bwr)
    plt.axvline(x=coordinates.loc['EPIC1', 'tSNE-1'], ls=':')
    plt.axhline(y=coordinates.loc['EPIC1', 'tSNE-2'], ls=':')
    plt.show()

    return


def visual_sub_tSNE(try_id, subset, label):
    coordinates = pd.read_csv(
        'corr_68sig_linc/tSNE_CR_score/tSNE_embedding_' + try_id + '.csv',
        header=0,
        index_col=0,
        sep=',')

    coordinates['tSNE-1'] = (coordinates['tSNE-1'] - coordinates['tSNE-1'].min()) / (coordinates['tSNE-1'].max() - coordinates['tSNE-1'].min())
    coordinates['tSNE-2'] = (coordinates['tSNE-2'] - coordinates['tSNE-2'].min()) / (coordinates['tSNE-2'].max() - coordinates['tSNE-2'].min())

    coordinates = df(coordinates.loc[subset.index, :])
    plt.subplots(figsize=(8, 8))
    if label is None:
        plt.scatter(
            coordinates['tSNE-1'], coordinates['tSNE-2'],
            s=20, c='grey', linewidths=0)
    else:
        plt.scatter(
            coordinates['tSNE-1'], coordinates['tSNE-2'],
            s=20, c=subset[label], linewidths=0,
            vmin=-1, vmax=1, cmap=plt.cm.bwr)
    plt.axvline(x=coordinates.loc['EPIC1', 'tSNE-1'], ls=':')
    plt.axhline(y=coordinates.loc['EPIC1', 'tSNE-2'], ls=':')
    plt.show()

    return


def run_AP(try_id, data):
    clustering = AffinityPropagation().fit(data)
    label_lncRNAs = df(index=data.index, columns=['label_assigned'])
    label_lncRNAs['label_assigned'] = clustering.labels_
    label_lncRNAs.to_csv('corr_68sig_linc/tSNE_CR_score/clustering/AP_' + try_id + '.csv', sep=',')

    return label_lncRNAs



def run_DBSCAN(try_id, subset, eps, min_samples):
    # read in tSNE embedding coordinates
    coordinates = pd.read_csv(
        'corr_68sig_linc/tSNE_CR_score/tSNE_embedding_' + try_id + '.csv',
        header=0,
        index_col=0,
        sep=',')

    if subset != None:
        coordinates = df(coordinates.loc[subset.index, :])

    # scaled to [0, 1]
    coordinates['tSNE-1'] = (coordinates['tSNE-1'] - coordinates['tSNE-1'].min()) / (coordinates['tSNE-1'].max() - coordinates['tSNE-1'].min())
    coordinates['tSNE-2'] = (coordinates['tSNE-2'] - coordinates['tSNE-2'].min()) / (coordinates['tSNE-2'].max() - coordinates['tSNE-2'].min())

    # input hyperparameter
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)

    # initial assign
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    label_cell = df(index=coordinates.index, columns=['cluster'])
    label_cell['cluster'] = labels
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    # visualize
    plt.subplots(figsize=(10, 10))
    plt.scatter(coordinates['tSNE-1'], coordinates['tSNE-2'], c=label_cell['cluster'], s=20, linewidths=0, cmap='Dark2')
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.axvline(x=coordinates.loc['EPIC1', 'tSNE-1'], ls=':')
    plt.axhline(y=coordinates.loc['EPIC1', 'tSNE-2'], ls=':')
    plt.show()
    print('EPIC1 is in ' + str(label_cell.loc['EPIC1', :]))
    
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(coordinates, labels))
    
    return label_cell


def number_cluster(label_cell):
    # show number of genes in each cluster
    for c in label_cell['cluster'].unique():
        print('cluster ' + str(c))
        print(len(label_cell[label_cell['cluster'] == c].index))
    return



def report_KNN(results, n_top, try_id):
    f = open('corr_68sig_linc/classifier/' + try_id + '_KNN_hyper_parameter_selection.txt', 'w')
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            f.write("Model with rank: {0}".format(i))
            f.write('\n')
            f.write("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
            f.write('\n')
            f.write("Parameters: {0}".format(results['params'][candidate]))
            f.write('\n')
            f.write("")
    return


def hyperpara_KNN(target, training, try_id):
    Y = target[target['cluster'] != -1]
    X = df(training.loc[Y.index, :])
    
    # select KNN for the following training
    clf = KNeighborsClassifier(p=2)

    # specify parameters and distributions to sample from
    param_dist = {"n_neighbors": np.arange(5, 50, 5),
                  "leaf_size": np.arange(30, 80, 5),
                  "weights": ['uniform', 'distance']}

    # run randomized search
    n_iter_search = 50
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5,
                                       random_state=0,
                                       refit=True)

    random_search.fit(np.array(X.values), np.ravel(Y.values))
    report_KNN(results=random_search.cv_results_, n_top=10, try_id=try_id)

    return


def final_KNN(target, training, n_neighbours, leaf_size, weights):
    Y = target[target['cluster'] != -1]
    X = df(training.loc[Y.index, :])

    # construction of final model
    clf = KNeighborsClassifier(
        n_neighbors=n_neighbours,
        leaf_size=leaf_size,
        weights=weights,
        p=2)

    class_label = Y['cluster'].unique()
    class_label.sort()

    # evaluate by 5-fold cross-validation
    score = cross_val_score(clf, np.array(X.values), np.ravel(Y.values), cv=5)

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(np.array(X.values)):
        X_train, X_test = np.array(X.values)[train_index], np.array(X.values)[test_index]
        y_train, y_test = np.ravel(Y.values)[train_index], np.ravel(Y.values)[test_index]
        clf_fit = clf.fit(X_train, y_train)
        y_true = np.ravel(y_test)
        y_pred = clf_fit.predict(X_test)
        cmatrix = confusion_matrix(y_true, y_pred, labels=class_label)
        cmatrix = cmatrix.astype(float) / cmatrix.sum(axis=1)[:, np.newaxis]
        cmatrix_frame = df(cmatrix, index=class_label, columns=class_label)

    # visualize the confusion matrix
    sns.heatmap(cmatrix)
    plt.show()
    
    # prediction
    X_pred = df(training.loc[target[target['cluster'] == -1].index, :])
    sample_label = X_pred.index
    ambi_predict = clf.predict_proba(X_pred.values)
    proba_frame = df(ambi_predict, index=sample_label, columns=class_label)

    # only get assignment with posterior probability >= .99
    proba_frame = proba_frame[proba_frame >= .99]

    # assign the predicted cell type
    posterior_assign = target.copy()
    for c in proba_frame.columns:
        current_assign = proba_frame[c].dropna(axis=0, how='any').index
        posterior_assign.at[current_assign, 'cluster'] = c
    
    return posterior_assign


def visual_posterior_assign(try_id, subset, posterior_assign):
    # read in tSNE embedding coordinates
    coordinates = pd.read_csv(
        'corr_68sig_linc/tSNE_CR_score/tSNE_embedding_' + try_id + '.csv',
        header=0,
        index_col=0,
        sep=',')

    if subset is not None:
        coordinates = df(coordinates.loc[subset.index, :])
    # visualize
    plt.subplots(figsize=(10, 10))
    plt.scatter(coordinates['tSNE-1'], coordinates['tSNE-2'], c=posterior_assign['cluster'], s=20, linewidths=0, cmap='Dark2')
    plt.title('Posterior assign by KNN')
    plt.axvline(x=coordinates.loc['EPIC1', 'tSNE-1'], ls=':')
    plt.axhline(y=coordinates.loc['EPIC1', 'tSNE-2'], ls=':')
    plt.show()
    
    return

def visual_cluster(posterior_assign, data):
    for c in posterior_assign['cluster'].unique():
        if c == -1:
            continue
        print('cluster ' + str(c))
        tmp_linc = posterior_assign[posterior_assign['cluster'] == c].index
        sns.clustermap(
            data.loc[tmp_linc, :].astype(float),
            figsize=(16, 10),
            cmap='bwr', vmin=-8, vmax=8)
        plt.show()
    return


# # Analyses

# ## 1. Read consensus regulation score
# Concensus regulation score (CR score) is defined by positive hits minus negative hits. The range of CR score is -30 to 30, indicating the regulation direction is consensus negative or consensus positive.

# In[3]:


CR_score = pd.read_csv(
    'corr_68sig_linc/dich/pos_neg_immune_count.csv', 
    header=0, index_col=0, sep=',')
positive = pd.read_csv(
    'corr_68sig_linc/dich/promote_immune_count.csv', 
    header=0, index_col=0, sep=',')
negative = pd.read_csv(
    'corr_68sig_linc/dich/inhibit_immune_count.csv', 
    header=0, index_col=0, sep=',')


# In[27]:


# mean distribution of CR_score
sns.kdeplot(CR_score.mean(axis=1))


# In[11]:


# variance distribution of CR_score
sns.kdeplot(CR_score.std(axis=0))


# ## 2. The sign distribution of CR score

# In[14]:


# get the sign of CR score
sign_CR_score = df(np.sign(CR_score))


# In[15]:


# visualize by heatmap
sns.clustermap(sign_CR_score, cmap='bwr', vmin=-1, vmax=1)


# ## 3. tSNE on sign of CR score
# To determine if the sign will tell something.

# In[17]:


# run tSNE
run_tSNE(data=sign_CR_score, try_id='sign_of_CR')


# In[18]:


# visualize tSNE in MHC
visual_tSNE(try_id='sign_of_CR', data=sign_CR_score, label='MHC.I_19272155')


# In[24]:


# visualize tSNE in Module 11 proliferation
visual_tSNE(try_id='sign_of_CR', data=sign_CR_score, label='Module11_Prolif_score')


# In[22]:


# visualize tSNE (rw embedding)
visual_tSNE(try_id='sign_of_CR', data=sign_CR_score, label=None)


# ## 3. tSNE on CR score

# ### 3.1 tSNE on full set

# In[25]:


# run tSNE
run_tSNE(data=CR_score, try_id='raw_CR')


# In[28]:


# visualize tSNE in MHC
visual_tSNE(try_id='raw_CR', data=CR_score, label=None)


# In[29]:


# run tSNE
run_tSNE(data=CR_score / 32., try_id='scaled_CR')


# In[143]:


visual_tSNE(try_id='scaled_CR', data=CR_score, label=None)


# In[144]:


visual_tSNE(try_id='scaled_CR', data=CR_score, label='MHC.I_19272155')


# ### 3.2 tSNE on reduced set
# Only perform tSNE on lncRNAs with CR score > 5 in at least 10 signature

# In[22]:


# number of lncRNAs as CR cutoff goes high
lncRNA_CR_cutoff = df(columns=['CR_cutoff', '#lncRNAs'])
lncRNA_CR_sig = {}
for i in range(0, int(abs(CR_score.max().max()))):
    lncRNA_CR_cutoff.at[i, 'CR_cutoff'] = i
    tmp_CR = abs(CR_score.copy())
    tmp_CR = df(tmp_CR[tmp_CR >= i]).dropna(axis=0, how='all')
    lncRNA_CR_cutoff.at[i, '#lncRNAs'] = len(tmp_CR.index)

    tmp_count = df(columns=['sig_cutoff', '#lncRNAs'])
    for j in range(0, 68):
        lead_lncRNA = 0
        for lnc in tmp_CR.index:
            if len(df(tmp_CR.loc[lnc, :]).dropna(axis=0, how='any').index) >= j:
                lead_lncRNA += 1
            else:
                continue
        tmp_count.at[j, 'sig_cutoff'] = j
        tmp_count.at[j, '#lncRNAs'] = lead_lncRNA
    lncRNA_CR_sig[i] = tmp_count
    


# In[44]:


plt.figure(figsize=(8, 8))
sns.set_palette(palette='Spectral', n_colors=32)
for k in lncRNA_CR_sig.keys():
    sns.scatterplot(x='sig_cutoff', y='#lncRNAs', data=lncRNA_CR_sig[k])
plt.yticks(rotation=90)
plt.tight_layout()
plt.axvline(x=10., ls=':', color='k')
plt.show()


# In[81]:


# number of lncRNAs as CR cutoff goes high
lncRNA_CR_fraction = df(columns=['CR_cutoff', '#lncRNAs/sig'])
for i in range(0, int(abs(CR_score.max().max()))):
    lncRNA_CR_fraction.at[i, 'CR_cutoff'] = i
    lncRNA_CR_fraction.at[i, '#lncRNAs/sig'] = df(CR_score[abs(CR_score) < i]).isna().sum().mean() / len(CR_score.index)
    


# In[100]:


plt.figure(figsize=(6, 6))
sns.scatterplot(x='CR_cutoff', y='#lncRNAs/sig', data=lncRNA_CR_fraction)
plt.axvline(x=5., ls=':', color='k')


# In[112]:


sns.kdeplot(CR_score.mean(axis=0))
plt.axvline(x=5., ls=':', color='k')
plt.axvline(x=-5., ls=':', color='k')
print(CR_score.mean(axis=0).mean() + CR_score.mean(axis=0).std()*2)


# In[110]:


sns.kdeplot(CR_score.mean(axis=1))
plt.axvline(x=10., ls=':', color='k')
plt.axvline(x=-10., ls=':', color='k')
print(CR_score.mean(axis=1).std()*2)


# In[89]:


# number of lncRNAs as CR cutoff goes high
lncRNA_CR_sig_fraction = df(columns=['CR_cutoff', '#sig/lncRNA'])
for i in range(0, int(abs(CR_score.max().max()))):
    lncRNA_CR_sig_fraction.at[i, 'CR_cutoff'] = i
    tmp_cr = df(CR_score[abs(CR_score) < i]).isna().sum(axis=1).mean()
    lncRNA_CR_sig_fraction.at[i, '#sig/lncRNA'] = tmp_cr


# In[90]:


plt.figure(figsize=(6, 6))
sns.scatterplot(x='CR_cutoff', y='#sig/lncRNA', data=lncRNA_CR_sig_fraction)


# In[16]:


print(lncRNA_CR_cutoff)


# In[17]:


sns.scatterplot(x='CR_cutoff', y='#lncRNAs', data=lncRNA_CR_cutoff)


# In[48]:


# lncRNAs with CR > 5 at least in one signature
rr_CRS = abs(CR_score.copy())
rr_CRS = df(rr_CRS[rr_CRS >= 5.]).dropna(axis=0, how='all')
print(rr_CRS.shape)


# In[16]:


sns.kdeplot(rr_CRS.isna().sum()/len(rr_CRS['MHC.I_19272155']))


# In[30]:


(CR_score[CR_score == 0].isna().sum()).mean() / len(CR_score.index)


# In[32]:


(rr_CRS.isna().sum()).mean() / len(CR_score.index)


# In[28]:


sns.kdeplot(rr_CRS.isna().sum()/len(CR_score['MHC.I_19272155']))


# In[15]:


df(rr_CRS.isna().sum()/len(rr_CRS['MHC.I_19272155'])).mean()


# In[62]:


# lncRNAs with CR > 5 at least in ten signature
lead_lncRNA = []
for lnc in rr_CRS.index:
    if len(df(rr_CRS.loc[lnc, :]).dropna(axis=0, how='any').index) >= 10:
        lead_lncRNA.append(lnc)
    else:
        continue
print(len(lead_lncRNA))
rr_CRS = df(rr_CRS.loc[lead_lncRNA, :])


# In[63]:


df(rr_CRS.isna().sum()/len(CR_score.index)).mean()


# In[96]:


print(((len(rr_CRS.index) - rr_CRS.isna().sum()) / len(CR_score.index)).mean())


# In[72]:


print(df(CR_score[abs(CR_score) < 5]).isna().sum().mean() / len(CR_score.index))


# In[17]:


4292 / 6569


# In[18]:


print(CR_score.shape)


# In[120]:


# attach the sign
rr_CRS = np.sign(CR_score) * rr_CRS


# In[123]:


rr_CRS = rr_CRS.dropna(axis=0, how='all').fillna(0)


# In[125]:


print(rr_CRS.loc['EPIC1', :].head())


# ### 3.2.1 Reduced set with lower value masked (CR score < 5 will be masked)

# In[129]:


# run tSNE
run_tSNE(data=rr_CRS, try_id='reduced_set_CR')

# visualize
visual_tSNE(try_id='reduced_set_CR', data=rr_CRS, label=None)


# In[131]:


# visualize
visual_tSNE(try_id='reduced_set_CR', data=rr_CRS, label='MHC.I_19272155')


# In[132]:


# visualize
visual_tSNE(try_id='reduced_set_CR', data=rr_CRS, label='Module11_Prolif_score')


# In[133]:


sns.clustermap(rr_CRS, cmap='bwr', vmin=-10, vmax=10)


# In[137]:


sns.scatterplot(x='Module11_Prolif_score', y='MHC.I_19272155', data=CR_score)


# In[138]:


sns.scatterplot(x='Module11_Prolif_score', y='MHC.I_19272155', data=rr_CRS)


# ### 3.2.2 Reduced set with lower value unmasked (CR score < 5 will not be masked)

# In[317]:


# run tSNE
run_tSNE(data=CR_score.loc[rr_CRS.index, :], try_id='reduced_CR_low_unmask')

# visualize
visual_tSNE(try_id='reduced_CR_low_unmask', data=rr_CRS, label=None)


# In[320]:


# visualize the unmasked one
visual_tSNE(try_id='reduced_CR_low_unmask', data=CR_score.loc[rr_CRS.index, :], label='MHC.I_19272155')


# In[321]:


visual_tSNE(try_id='reduced_CR_low_unmask', data=CR_score.loc[rr_CRS.index, :], label='Module11_Prolif_score')


# ### 3.3 Plot the reduced set in the full embedding

# In[153]:


# only plot the reduced set on the full set tSNE
visual_sub_tSNE(try_id='scaled_CR', subset=rr_CRS, label=None)


# In[154]:


visual_sub_tSNE(try_id='scaled_CR', subset=rr_CRS, label='MHC.I_19272155')


# In[149]:


visual_sub_tSNE(try_id='scaled_CR', subset=rr_CRS, label='Module11_Prolif_score')


# ## 4. DBSCAN on tSNE embedding
# Density-based spatial clustering of applications with noise (DBSCAN) is a data clustering algorithm proposed by Martin Ester, Hans-Peter Kriegel, Jörg Sander and Xiaowei Xu in 1996.It is a density-based clustering algorithm: given a set of points in some space, it groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away). DBSCAN is one of the most common clustering algorithms and also most cited in scientific literature.

# ### 4.1 Clustering on embedding with reduced set (low value  masked)

# In[427]:


# initial assign
label = run_DBSCAN(try_id='reduced_set_CR', subset=None, eps=.08, min_samples=180)
number_cluster(label_cell=label)


# In[258]:


# posterior assign
hyperpara_KNN(target=label, training=rr_CRS, try_id='reduced_set_CR')


# In[265]:


# final model building
posterior_assign = final_KNN(target=label, training=rr_CRS, n_neighbours=10, leaf_size=75, weights='distance')


# In[270]:


# visual final assign
visual_posterior_assign(try_id='reduced_set_CR', subset=None, posterior_assign=posterior_assign)
number_cluster(label_cell=posterior_assign)


# In[316]:


visual_cluster(posterior_assign=posterior_assign, data=CR_score.loc[rr_CRS.index, :])


# In[322]:


# compare clusters: average CR score
average_CR = df()
for c in [0, 3, 2, 1]:
    tmp_linc = posterior_assign[posterior_assign['cluster'] == c].index
    average_CR[c] = CR_score.loc[tmp_linc, :].mean()
sns.clustermap(average_CR, cmap='bwr', z_score=0,
               figsize=(8, 16), col_cluster=False)


# In[408]:


# EPIC1 in cluster 0
cluster_0 = rr_CRS.loc[posterior_assign[posterior_assign['cluster'] == 0].index, :]

# sort by module 11 and MHCI
mhc1 = df(cluster_0[cluster_0['MHC1_21978456'] <= -8])
mhc1 = df(mhc1[mhc1['MHC.I_19272155'] <= -8])
prolif = df(cluster_0[cluster_0['Module11_Prolif_score'] >= 8])
op_mhc1_prolif = list(set(mhc1.index) & set(prolif.index))
print(len(op_mhc1_prolif))


# In[324]:


sns.clustermap(CR_score.loc[op_mhc1_prolif, :], cmap='bwr', figsize=(16, 16), vmin=-10, vmax=10)


# ### 4.2 Clustering on embedding with reduced set (low value not masked)

# In[429]:


# initial assign
label_unmasked = run_DBSCAN(try_id='reduced_CR_low_unmask', subset=None, eps=.077, min_samples=150)
number_cluster(label_cell=label_unmasked)


# In[368]:


# posterior assign
rr_un_CRS = df(CR_score.loc[rr_CRS.index, :])
hyperpara_KNN(target=label_unmasked, training=rr_un_CRS, try_id='reduced_CR_low_unmask')


# In[369]:


# final model building
posterior_assign_unmasked = final_KNN(target=label_unmasked, training=rr_un_CRS, n_neighbours=10, leaf_size=75, weights='distance')


# In[422]:


# visual final assign
visual_posterior_assign(try_id='reduced_CR_low_unmask', subset=None, posterior_assign=posterior_assign_unmasked)
number_cluster(label_cell=posterior_assign_unmasked)


# In[418]:


# write results
posterior_assign_unmasked.to_csv('corr_68sig_linc/CR_clusters/posterior_assign_unmasked.csv', sep=',')
rr_un_CRS.to_csv('corr_68sig_linc/CR_clusters/unmasked_reduced_CR_score.csv', sep=',')


# In[371]:


visual_cluster(posterior_assign=posterior_assign_unmasked, data=rr_un_CRS)


# In[421]:


# compare clusters: average CR score
average_CR_unmasked = df()
for c in [0, 2, 1, 3]:
    tmp_linc = posterior_assign_unmasked[posterior_assign_unmasked['cluster'] == c].index
    average_CR_unmasked[c] = CR_score.loc[tmp_linc, :].mean()
sns.clustermap(average_CR_unmasked, cmap='bwr', z_score=0,
               figsize=(8, 16), col_cluster=False)
plt.subplots_adjust(left=0.1, right=0.7)
fig = plt.gcf()
fig.savefig('corr_68sig_linc/CR_clusters/average_CR_score_across_clusters.png', dpi=150)


# In[430]:


print(rr_un_CRS.loc['EPIC1', :])


# In[404]:


# EPIC1 in cluster 1
cluster_1 = rr_un_CRS.loc[posterior_assign_unmasked[posterior_assign_unmasked['cluster'] == 1].index, :]

# sort by module 11 and MHCI: regulate 25% of the cancer types (8)
mhc1_1 = df(cluster_1[cluster_1['MHC1_21978456'] <= -8])
mhc1_1 = df(mhc1_1[mhc1_1['MHC.I_19272155'] <= -8])
prolif_1 = df(cluster_1[cluster_1['Module11_Prolif_score'] >= 8])
op_mhc1_prolif_1 = list(set(mhc1_1.index) & set(prolif_1.index))
print(len(op_mhc1_prolif_1))


# In[407]:


sns.clustermap(CR_score.loc[op_mhc1_prolif_1, :], cmap='bwr', figsize=(16, 4), vmin=-8, vmax=8)


# In[409]:


# cancer type expression
wilcox = pd.read_csv(
    'corr_68sig_linc/DEG/wilcoxon_ranksum_T_vs_N_lincRNAs.csv',
    index_col=0, header=0, sep=',')


# In[414]:


sns.clustermap(wilcox.loc[op_mhc1_prolif_1, :], cmap='bwr', vmin=-10, vmax=10, figsize=(10, 4))


# In[417]:


df(wilcox.loc[op_mhc1_prolif_1, :].mean(axis=1)).sort_values(by=0, ascending=False)


# In[432]:


sns.clustermap(wilcox.loc[df(wilcox.loc[op_mhc1_prolif_1, :].mean(axis=1)).sort_values(by=0, ascending=False).index, :], row_cluster=False, cmap='bwr', vmin=-8, vmax=8, figsize=(10, 4))


# In[433]:


sns.clustermap(CR_score.loc[df(wilcox.loc[op_mhc1_prolif_1, :].mean(axis=1)).sort_values(by=0, ascending=False).index, :], row_cluster=False, cmap='bwr', vmin=-8, vmax=8, figsize=(10, 4))


# In[442]:


ranked_list = df(wilcox.loc[op_mhc1_prolif_1, :].mean(axis=1)).sort_values(by=0, ascending=False)
plt.figure(figsize=(2, 8))
sns.barplot(x=ranked_list[0], y=ranked_list.index, color='gray')
plt.xlabel('Average wilcoxon p value of tumor versus normal')


# ## 5. Clustering on CR score
# tSNE may not be the best way to segregate the sample due to the big variance. Let's try clustering.

# ### 5.1 Affinity Propagation
# In statistics and data mining, affinity propagation (AP) is a clustering algorithm based on the concept of "message passing" between data points.
# Unlike clustering algorithms such as k-means or k-medoids, affinity propagation does not require the number of clusters to be determined or estimated before running the algorithm. Similar to k-medoids, affinity propagation finds "exemplars", members of the input set that are representative of clusters.

# In[35]:


# first try
labels = run_AP(data=CR_score, try_id='raw_CR')


# In[38]:


print(len(labels['label_assigned'].unique()))


# ### 5.2 hierarchical clustering

# In[41]:


sns.clustermap(CR_score, cmap='bwr', vmin=-10, vmax=10)


# ### 5.3 K-means
# k-means clustering is a method of vector quantization, originally from signal processing, that is popular for cluster analysis in data mining. k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells.

# #### 5.3.1 PCA on the original space

# In[47]:


PCA_CRS = PCA(n_components=2).fit(CR_score)
print(PCA_CRS.explained_variance_ratio_)


# In[48]:


reduced_CRS = PCA(n_components=2).fit_transform(CR_score)


# In[66]:


plt.figure(figsize=(8, 8))
plt.scatter(x=reduced_CRS[:, 0], y=reduced_CRS[:, 1], s=10, alpha=0.5)

