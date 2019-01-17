
# coding: utf-8

# # Imported Modules

# In[359]:


import matplotlib
import os
import sys

import numpy as np
import scipy.stats as stat
import pandas as pd
import seaborn as sns
import statsmodels.stats.multitest as ssm

from pandas import DataFrame as df
from matplotlib import pyplot as plt

import matplotlib.font_manager as fm
from matplotlib.ft2font import FT2Font

def get_font(*args, **kwargs):
    return FT2Font(*args, **kwargs)

fm.get_font = get_font


# # All functions

# In[363]:


def sub_cluster(posterior_assign, data, z_score):
    reordered_lnc = []
    for c in ['inhibit_im_cancer_shared','inhibit_im_cancer_specific', 'promote_im_cancer_specific', 'promote_im_cancer_shared']:
        print('cluster ' + str(c))
        tmp_linc = posterior_assign[posterior_assign['assigned'] == c].index
        g = sns.clustermap(
            data.loc[tmp_linc, :].astype(float),
            row_cluster=True, col_cluster=False, z_score=z_score)
        plt.close()
        reordered_lnc.extend([tmp_linc[i] for i in g.dendrogram_row.reordered_ind])

    return reordered_lnc


def sub_distribution(posterior_assign, data):
    color_cluster = {'noise': 'lightgray',
                 'inhibit_im_cancer_specific': 'lightseagreen',
                 'inhibit_im_cancer_shared': 'royalblue',
                 'promote_im_cancer_specific': 'coral',
                 'promote_im_cancer_shared': 'crimson'}
    tmp_linc = posterior_assign[posterior_assign['assigned'] != 'noise'].index
    data = data.loc[tmp_linc, :]
    for col in data.columns:
        print(col)
        sns.boxplot(
            y=data[col], x=posterior_assign['assigned'],
            order=['inhibit_im_cancer_shared','inhibit_im_cancer_specific', 'promote_im_cancer_specific', 'promote_im_cancer_shared'],
            palette=color_cluster)
        plt.axhline(y=0., ls=':', c='k')
        plt.axhline(y=2., ls=':', c='r')
        plt.axhline(y=-2., ls=':', c='b')
        plt.show()
    return


def get_corr(gene, biotype):
    '''
    Get the correlation and FDR (adjusted p value) from source.
    Parameter:
        gene: str, name of the gene
        biotype: str, ['linc', 'PCG']
    Return:
        corr_matrix, fdr_matrix: dataframe, the correlation and FDR matrix of the given gene.
    '''
    cancer_type = os.listdir('/home/yuew/Documents/TCGA/GDC/')
    corr_matrix = df(columns=cancer_type)
    fdr_matrix = df(columns=cancer_type)
    
    for ct in cancer_type:
        corr_matrix[ct] = pd.read_csv(
            'corr_68sig_' + biotype + '/corr/spearmanr_coef_' + ct + '.csv',
            header=0, index_col=0, sep=',').loc[gene, :]
        fdr_matrix[ct] = pd.read_csv(
            'corr_68sig_' + biotype + '/fdr/spearmanr_fdr_' + ct + '.csv',
            header=0, index_col=0, sep=',').loc[gene, :]
    
    return corr_matrix, fdr_matrix


def boPlot(corr, fdr):
    '''
    Visualize by bo plot. Dot size indicate the significance, color indicate the correlation
    '''
    location = {}
    j = 1
    for ctype in fdr.columns:
        location[ctype] = df(index=fdr.index, columns=['x', 'y'])
        location[ctype]['fdr'] = fdr[ctype]
        location[ctype]['corr'] = corr[ctype]

        i = 1
        for t in location[ctype].index:
            location[ctype].at[t, 'y'] = i - .5
            location[ctype].at[t, 'x'] = j - .5
            i += 1
        j += 1

    # collapse all subframe to one
    mergeIndex = []
    for c in location.keys():
        mergeIndex.extend([c + '~' + x for x in location[c].index])
    mergeLoc = df(index=mergeIndex, columns=['x', 'y', 'fdr', 'corr'])
    for c in location.keys():
        for t in fdr.index:
            mergeLoc.at[c + '~' + t, 'x'] = location[c].loc[t, 'x']
            mergeLoc.at[c + '~' + t, 'y'] = location[c].loc[t, 'y']
            mergeLoc.at[c + '~' + t, 'fdr'] = location[c].loc[t, 'fdr']
            mergeLoc.at[c + '~' + t, 'corr'] = location[c].loc[t, 'corr']
    
    # remove points with FDR < 0.1
    mergeLoc = mergeLoc[mergeLoc['fdr'] >= 1].dropna(axis=0, how='any')
    
    fig = plt.figure(figsize=(8, 32))
    ax = fig.add_subplot(111, aspect='equal')
    sns.heatmap(corr, vmin=-.25, vmax=.25, cmap='bwr', square=True, linewidths=0.05, xticklabels=True, yticklabels=True)

    # sns.scatterplot(x=mergeLoc['x'], y=mergeLoc['y'], size=mergeLoc['value'], color='k', linewidths=False)
    ax.scatter(
        x=mergeLoc['x'], y=mergeLoc['y'],
        s=mergeLoc['fdr'].astype(float)*20, color='k',
        # c=mergeLoc['corr'], cmap=plt.cm.bwr, vmin=-.4, vmax=.4,
        )
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    
    plt.subplots_adjust(left=0.2)
    
    # legend
    l1 = plt.scatter([],[], s=20, edgecolors='none', c='k') # FDR = 10e-1
    l2 = plt.scatter([],[], s=60, edgecolors='none', c='k') # FDR = 10e-3
    l3 = plt.scatter([],[], s=100, edgecolors='none', c='k') # FDR = 10e-5
    l4 = plt.scatter([],[], s=140, edgecolors='none', c='k') # FDR = 10e-7

    labels = ["10e-1", "10e-3", "10e-5", "10e-7"]

    leg = plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True, fontsize=12)


    
    fig = plt.gcf()
    fig.savefig('figures/vectors/F2/1a_EPIC1_TCGA.pdf', transparent=True)
    plt.show()

    return


def boPlot_gsea(corr, fdr):
    '''
    Visualize by bo plot for GSEA. Dot size indicate the significance, color indicate the correlation
    '''
    location = {}
    j = 1
    for ctype in fdr.columns:
        location[ctype] = df(index=fdr.index, columns=['x', 'y'])
        location[ctype]['fdr'] = fdr[ctype]
        location[ctype]['corr'] = corr[ctype]

        i = 1
        for t in location[ctype].index:
            location[ctype].at[t, 'y'] = i - .5
            location[ctype].at[t, 'x'] = j - .5
            i += 1
        j += 1

    # collapse all subframe to one
    mergeIndex = []
    for c in location.keys():
        mergeIndex.extend([c + '~' + x for x in location[c].index])
    mergeLoc = df(index=mergeIndex, columns=['x', 'y', 'fdr', 'corr'])
    for c in location.keys():
        for t in fdr.index:
            mergeLoc.at[c + '~' + t, 'x'] = location[c].loc[t, 'x']
            mergeLoc.at[c + '~' + t, 'y'] = location[c].loc[t, 'y']
            mergeLoc.at[c + '~' + t, 'fdr'] = location[c].loc[t, 'fdr']
            mergeLoc.at[c + '~' + t, 'corr'] = location[c].loc[t, 'corr']
    
    # remove points with p > 0.1
    mergeLoc = mergeLoc[mergeLoc['fdr'] >= 1].dropna(axis=0, how='any')
    
    fig = plt.figure(figsize=(6, 24))
    ax = fig.add_subplot(111, aspect='equal')
    sns.heatmap(corr * 0, vmin=-2, vmax=2, cmap='bwr', square=True, linewidths=0.05, linecolor='k', xticklabels=True, yticklabels=True)

    # sns.scatterplot(x=mergeLoc['x'], y=mergeLoc['y'], size=mergeLoc['value'], color='k', linewidths=False)
    ax.scatter(
        x=mergeLoc['x'], y=mergeLoc['y'],
        s=mergeLoc['fdr'].astype(float)*40,
        # color='k',
        c=mergeLoc['corr'], cmap=plt.cm.bwr, vmin=-2, vmax=2,
        )
    plt.xticks(rotation=90, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    
    plt.subplots_adjust(left=0.2)
    
    # legend
    l1 = plt.scatter([],[], s=40, edgecolors='none', c='k') # FDR = 10e-1
    l2 = plt.scatter([],[], s=80, edgecolors='none', c='k') # FDR = 10e-2
    l3 = plt.scatter([],[], s=120, edgecolors='none', c='k') # FDR = 10e-3
    l4 = plt.scatter([],[], s=160, edgecolors='none', c='k') # FDR = 10e-4
    labels = ["10e-1", "10e-2", "10e-3", "10e-4"]

    leg = plt.legend([l1, l2, l3, l4], labels, ncol=4, frameon=True, fontsize=12)
    fig = plt.gcf()
    fig.savefig('figures/vectors/F2/1a_EPIC1_siRNA.pdf', transparent=True)
    plt.show()

    return


def corr_pcg_lnc(pcg_list, lnc_list, ctype, convert_id):
    '''
    Calculate the correlation between the given list of protein coding genes and lncRNAs in given cancer type.
    '''
    if ctype == 'TCGA-SKCM':
        tmp_expr = pd.read_csv(
            '/home/yuew/Documents/TCGA/GDC/' + ctype + '/' + ctype + '_metastasis_patient_expression.csv',
            header=0, index_col=0, sep=',').rename(index=convert_id['gene_symbol'])
    else:
        tmp_expr = pd.read_csv(
            '/home/yuew/Documents/TCGA/GDC/' + ctype + '/' + ctype + '_primary_patient_expression.csv',
            header=0, index_col=0, sep=',').rename(index=convert_id['gene_symbol'])
    
    # get submatrix
    lnc_sub = df(tmp_expr.loc[lnc_list, :]).T
    pcg_sub = df(tmp_expr.loc[pcg_list, :]).T
    
    spr_corr = df(index=pcg_sub.columns, columns=lnc_sub.columns)
    spr_p = spr_corr.copy()
    
    for lnc in lnc_sub.columns:
        for pg in pcg_sub.columns:
            try:
                corr, p = stat.spearmanr(lnc_sub[lnc], pcg_sub[pg])
                spr_corr.at[pg, lnc] = corr
                spr_p.at[pg, lnc] = p
            except ValueError:
                pass
    
    adj_p = spr_p.copy()
    adj_bool = spr_p.copy()
    for col in adj_p.columns:
        adj_bool[col], adj_p[col] = ssm.fdrcorrection(adj_p[col], alpha=0.1, method='indep', is_sorted=False)
    fdr_bool = adj_bool * 1
    fdr_bool = fdr_bool.astype(float) * np.sign(spr_corr.astype(float))

    return spr_corr, spr_p, fdr_bool


# # Read-in

# In[69]:


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# In[70]:


# cancer type (n > 200)
ctype_200 = ['TCGA-LIHC', 'TCGA-LGG', 'TCGA-STAD', 'TCGA-BRCA', 'TCGA-HNSC', 'TCGA-UCEC', 'TCGA-BLCA', 'TCGA-LUAD', 'TCGA-COAD_READ', 'TCGA-KIRC', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-LUSC', 'TCGA-OV', 'TCGA-PRAD', 'TCGA-CESC', 'TCGA-KIRP', 'TCGA-THCA']
print(len(ctype_200))


# In[71]:


# read coordinations and cluster labels
coordinates = pd.read_csv(
        'corr_68sig_linc/tSNE_CR_score/tSNE_embedding_reduced_CR_low_unmask.csv',
        header=0,
        index_col=0,
        sep=',')

coordinates['tSNE-1'] = (coordinates['tSNE-1'] - coordinates['tSNE-1'].min()) / (coordinates['tSNE-1'].max() - coordinates['tSNE-1'].min())
coordinates['tSNE-2'] = (coordinates['tSNE-2'] - coordinates['tSNE-2'].min()) / (coordinates['tSNE-2'].max() - coordinates['tSNE-2'].min())

labels = pd.read_csv(
    'corr_68sig_linc/CR_clusters/posterior_assign_unmasked.csv',
    header=0, index_col=0, sep=',')

# read reduced CR score matrix
CR_score = pd.read_csv(
    'corr_68sig_linc/CR_clusters/unmasked_reduced_CR_score.csv',
    header=0, index_col=0, sep=',')


# In[72]:


# give name of clusters
name_cluster = {'-1': 'noise',
                '0': 'promote_im_cancer_specific',
                '1': 'inhibit_im_cancer_specific',
                '2': 'promote_im_cancer_shared',
                '3': 'inhibit_im_cancer_shared'}

# rename clusters
labels['assigned'] = labels['cluster'].astype(str)

for l in labels.index:
    labels.at[l, 'assigned'] = name_cluster[labels.loc[l, 'assigned']]

color_cluster = {'noise': 'lightgray',
                 'inhibit_im_cancer_specific': 'lightseagreen',
                 'inhibit_im_cancer_shared': 'royalblue',
                 'promote_im_cancer_specific': 'coral',
                 'promote_im_cancer_shared': 'crimson'}

marker_cluster = {'noise': 'X',
                 'inhibit_im_cancer_specific': 's',
                 'inhibit_im_cancer_shared': 'D',
                 'promote_im_cancer_specific': '^',
                 'promote_im_cancer_shared': 'o'}


# In[73]:


# all clusters
all_clusters = ['inhibit_im_cancer_shared','inhibit_im_cancer_specific', 'promote_im_cancer_specific', 'promote_im_cancer_shared']


# In[355]:


gene_name = pd.read_csv(
    '/home/yuew/Documents/Reference/GRCh38_ENSG_to_symbol.csv',
    header=0, index_col=0, sep=',')
gene_name.loc['ENSG00000224271', 'gene_symbol'] = 'EPIC1'


# In[75]:


# stromal fraction
stroma_corr = pd.read_csv(
    '/home/yuew/Documents/CARINA_DAUR/stromal_fraction/GDC_all_expr_CHAT_stromal_Fraction.csv',
    header=0, index_col=0, sep=',')
stroma_corr = stroma_corr.rename(index=gene_name['gene_symbol']).groupby(level=0).mean()
print(stroma_corr.shape)


# In[76]:


# CD8 infiltration
CD8T_corr = pd.read_csv(
    '/home/yuew/Documents/CARINA_DAUR/stromal_fraction/GDC_all_expr_TIMER_T_cell.CD8.csv',
    header=0, index_col=0, sep=',')
print(CD8T_corr.shape)
CD8T_corr = CD8T_corr.rename(index=gene_name['gene_symbol']).groupby(level=0).mean()


# In[189]:


# CD4 infiltration
CD4T_corr = pd.read_csv(
    '/home/yuew/Documents/CARINA_DAUR/stromal_fraction/GDC_all_expr_TIMER_T_cell.CD4.csv',
    header=0, index_col=0, sep=',')
print(CD4T_corr.shape)
CD4T_corr = CD4T_corr.rename(index=gene_name['gene_symbol']).groupby(level=0).mean()


# In[190]:


# macrophage
Macrophage_corr = pd.read_csv(
    '/home/yuew/Documents/CARINA_DAUR/stromal_fraction/GDC_all_expr_TIMER_Macrophage.csv',
    header=0, index_col=0, sep=',')
print(Macrophage_corr.shape)
Macrophage_corr = Macrophage_corr.rename(index=gene_name['gene_symbol']).groupby(level=0).mean()


# In[191]:


# DC
DC_corr = pd.read_csv(
    '/home/yuew/Documents/CARINA_DAUR/stromal_fraction/GDC_all_expr_TIMER_DC.csv',
    header=0, index_col=0, sep=',')
print(DC_corr.shape)
DC_corr = DC_corr.rename(index=gene_name['gene_symbol']).groupby(level=0).mean()


# In[192]:


# neutrophil
Neutrophil_corr = pd.read_csv(
    '/home/yuew/Documents/CARINA_DAUR/stromal_fraction/GDC_all_expr_TIMER_Neutrophil.csv',
    header=0, index_col=0, sep=',')
print(Neutrophil_corr.shape)
Neutrophil_corr = Neutrophil_corr.rename(index=gene_name['gene_symbol']).groupby(level=0).mean()


# In[193]:


# B cell
B_cell_corr = pd.read_csv(
    '/home/yuew/Documents/CARINA_DAUR/stromal_fraction/GDC_all_expr_TIMER_B_cell.csv',
    header=0, index_col=0, sep=',')
print(B_cell_corr.shape)
B_cell_corr = B_cell_corr.rename(index=gene_name['gene_symbol']).groupby(level=0).mean()


# In[86]:


# Human protein atlas
hpa = pd.read_csv(
    'expression_atlas/E-MTAB-2836-query-results.tpms.tsv',
    header=0, index_col=0, sep='\t').drop(['Gene Name'], axis=1).fillna(0.0)
log_hpa = np.log2(hpa + 1)
log_hpa = log_hpa.rename(gene_name['gene_symbol'])
print(log_hpa.shape)
op_lincRNA_hpa = list(set(log_hpa.index) & set(labels[labels['assigned'] != 'noise'].index))
print(len(op_lincRNA_hpa))


# In[87]:


# illumnia body map
ibm = pd.read_csv(
    'expression_atlas/E-MTAB-513-query-results.tpms.tsv',
    header=0, index_col=0, sep='\t').drop(['Gene Name'], axis=1).fillna(0.0)
log_ibm = np.log2(ibm + 1)
log_ibm = log_ibm.rename(gene_name['gene_symbol'])
print(log_ibm.shape)
op_lincRNA_ibm = list(set(log_ibm.index) & set(labels[labels['assigned'] != 'noise'].index))
print(len(op_lincRNA_ibm))


# In[147]:


# EPIC1 correlation and FDR with 68 immune signature
EPIC1_corr, EPIC1_fdr = get_corr(gene='EPIC1', biotype='linc')


# In[148]:


# merge NES results
cell_lines = os.listdir('RNA-seq/report/')
merge_cl_nes = df(columns=cell_lines)
merge_cl_fdr = df(columns=cell_lines)

for c in cell_lines:
    tmp_report = pd.read_csv(
        'RNA-seq/report/' + c + '/gseapy.prerank.gene_sets.report.csv',
        index_col=0, header=0, sep=',').dropna(axis=0, how='any')
    merge_cl_nes[c], merge_cl_fdr[c] = tmp_report['nes'], tmp_report['pval']


# In[153]:


ctype_200_selected = ['TCGA-LGG', 'TCGA-STAD', 'TCGA-BRCA', 'TCGA-HNSC', 'TCGA-BLCA', 'TCGA-LUAD', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-LUSC', 'TCGA-OV', 'TCGA-PRAD', 'TCGA-CESC', 'TCGA-KIRP']


# In[203]:


# probe list
lncRNA_probe = pd.read_csv(
    'CR_methylation/lncRNA_probe.csv',
    header=0, index_col=0, sep='\t')
# remove the dots in gene name
rn_id = {}
for g in lncRNA_probe.index:
    rn_id[g] = g.split('.')[0]
lncRNA_probe = lncRNA_probe.rename(index=rn_id)
print(lncRNA_probe.shape)

# rename the probe list use gene name
lncRNA_probe_gname = lncRNA_probe.rename(index=gene_name['gene_symbol'])
print(len(lncRNA_probe_gname.index))
print(len(lncRNA_probe_gname.index.unique()))

# merge the probes of each cluster together
probe_cluster = {}
for l in lncRNA_probe_gname.index:
    if l not in labels.index:
        continue
    if labels.loc[l, 'assigned'] in probe_cluster.keys():
        probe_cluster[labels.loc[l, 'assigned']].extend(str(lncRNA_probe_gname.loc[l, 'HM450 Probe ID']).split(','))
    else:
        probe_cluster[labels.loc[l, 'assigned']] = str(lncRNA_probe_gname.loc[l, 'HM450 Probe ID']).split(',')

# a reverse structure
probe_mapto_cluster = {}
for k in probe_cluster.keys():
    for l in probe_cluster[k]:
        probe_mapto_cluster[l] = k


# In[259]:


wilcox = pd.read_csv(
    'corr_68sig_linc/DEG/wilcoxon_ranksum_T_vs_N_lincRNAs.csv',
    index_col=0, header=0, sep=',')


# # _Figure 1_

# ## Fig. 1a: TSNE embedding of the four identified cancer-immunity-associated lncRNA clusters.
# ### Notes: 2019.1.11
# Put the legend besides the plot (right side). Rename the clusters into `cancer-specific` and `cancer-shared`

# In[145]:


plt.figure(figsize=(6, 6))
sns.scatterplot(
    x=coordinates['tSNE-1'], y=coordinates['tSNE-2'],
    hue=labels['assigned'], style=labels['assigned'],
    markers=marker_cluster,
    alpha=0.75, palette=color_cluster, linewidth=0.01)
plt.legend(prop={'size': 12})
plt.xlabel('TSNE-1', fontsize=20)
plt.ylabel('TSNE-2', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20, rotation=90)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/F1/1a_tsne_embedding.pdf', transparent=True)


# In[78]:


plt.close()


# ## Fig. 1b: Association between lncRNA expression and immune response signatures across cancers in different clusters.
# ### Notes: 2019.1.11
# Please add the color bar in the heatmap.

# In[79]:


clustered_lnc = sub_cluster(posterior_assign=labels, data=CR_score, z_score=False)
row_cluster_colors = labels['assigned'].map(color_cluster)
clustered_lnc_CR = df(CR_score.loc[clustered_lnc, :]).T

sns.clustermap(
    clustered_lnc_CR,
    col_cluster=False,
    vmin=-8, vmax=8, cmap='bwr',
    figsize=(10, 18), xticklabels=0,
    col_colors=row_cluster_colors, cbar_kws={'aspect': .50})

plt.subplots_adjust(left=0.05, right=0.7, top=0.99, bottom=0.05)
fig = plt.gcf()
fig.savefig('figures/vectors/F1/1b_clustermap.pdf', transparent=True)


# In[80]:


plt.close()


# ## Fig. 1c: Distribution of correlation between tumor stromal fractions and lncRNA expression in different clusters across TCGA cancer types.
# ### Notes: 2019.1.11
# Put the legend out of the plot. Exclude cancer types with sample size less than 200.
# Put the representative color of the clusters on the right side of the plot to indicate the identity of the clusters.

# In[81]:


sns.set_palette('tab20', 18, desat=1)
f, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
for i in range(len(all_clusters)):
    clt = all_clusters[i]
    plt.xlim(-.8, .8)
    tmp_linc = labels[labels['assigned'] == clt].index
    stroma_corr_tmp = stroma_corr.loc[tmp_linc, :]
    for col in stroma_corr_tmp.columns:
        if col in ctype_200:
            sns.kdeplot(stroma_corr_tmp[col].dropna(axis=0, how='any'), label=col, ax=axes[i], legend=False)
    plt.axvline(x=0.0, ls=':', c='k')
    plt.legend(loc='best', ncol=3, fontsize=8)
    plt.xlabel('Correlation with stromal fraction', fontsize=14)
    plt.ylabel('Density', fontsize=14)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/F1/1c_kde_stroma.pdf', transparent=True)


# In[82]:


plt.close()


# ## Fig. 1d: Distribution of correlation between tumor stromal fractions and lncRNA expression in different clusters across TCGA cancer types. 
# ### Notes: 2019.1.11
# Delete the legends (They are same as the one in stromal fraction)

# In[83]:


sns.set_palette('tab20', 18, desat=1)
f, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
for i in range(len(all_clusters)):
    clt = all_clusters[i]
    plt.xlim(-.8, .8)
    tmp_linc = labels[labels['assigned'] == clt].index
    CD8T_corr_tmp = CD8T_corr.loc[tmp_linc, :]
    for col in CD8T_corr_tmp.columns:
        if col in ctype_200:
            sns.kdeplot(CD8T_corr_tmp[col].dropna(axis=0, how='any'), label=col, ax=axes[i], legend=False)
    plt.axvline(x=0.0, ls=':', c='k')
    plt.legend(loc='best', ncol=3, fontsize=8)
    plt.xlabel('Correlation with CD8+ T cell infiltration', fontsize=14)
    plt.ylabel('Density', fontsize=14)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/F1/1c_kde_CD8T.pdf', transparent=True)


# In[84]:


plt.close()


# ## Fig. 1e: Expression of cancer-immunity-associated lncRNAs in immune-related tissues from healthy samples.
# ### Notes: 2019.1.11
# Remove `non-immune related tissues`. Add statistics on shared vs. specific comparison. Change the x-axis as `specific` and `shared`, put a `[` below showing it is `inhibiting` or `promoting`

# ### Human protein atlas

# In[88]:


lincRNA_hpa_expr = df(log_hpa.loc[op_lincRNA_hpa, :]).groupby(level=0).mean()
hpa_immune = ['lymph node', 'spleen', 'tonsil', 'bone marrow']

# barplot
merged_hpa = df(columns=['expression', 'cluster'])

for c in labels['assigned'].unique():
    if c == 'noise':
        continue
    
    # get subset of average expression
    tmp_expr = lincRNA_hpa_expr.loc[labels[labels['assigned'] == c].index, :]
    for ct in tmp_expr.columns:
        if ct in hpa_immune:
            merged_hpa.at[c + '|' + ct, 'expression'] = tmp_expr[ct].mean()
            merged_hpa.at[c + '|' + ct, 'cluster'] = c


# In[125]:


# barplot
merged_hpa_by_lnc = df(index=lincRNA_hpa_expr.index, columns=['expression', 'cluster'])

merged_hpa_by_lnc['expression'] = lincRNA_hpa_expr[hpa_immune].mean(axis=1)

for c in labels['assigned'].unique():
    if c == 'noise':
        continue  
    # get subset of average expression
    tmp_linc = list(set(labels[labels['assigned'] == c].index) & set(lincRNA_hpa_expr.index))
    
    merged_hpa_by_lnc.loc[tmp_linc, 'cluster'] = c


# In[140]:


plt.figure(figsize=(4, 4))
sns.barplot(x='cluster', y='expression', data=merged_hpa_by_lnc, order=all_clusters)
plt.xticks(rotation=90) # remember to change the name in AI
plt.yticks(rotation=90)
plt.ylabel('Log2FPKM') # significance: add ** in the figure indicating both of the p-values are less than 0.01
fig = plt.gcf()
fig.savefig('figures/vectors/F1/1d_hpa_immune.pdf', transparent=True)


# In[141]:


plt.close()


# In[127]:


# statistics: t-test
print(stat.ttest_ind(merged_hpa_by_lnc[merged_hpa_by_lnc['cluster'] == 'inhibit_im_cancer_shared']['expression'], merged_hpa_by_lnc[merged_hpa_by_lnc['cluster'] == 'inhibit_im_cancer_specific']['expression']))


# In[107]:


print(merged_hpa)


# In[128]:


print(stat.ttest_ind(merged_hpa_by_lnc[merged_hpa_by_lnc['cluster'] == 'promote_im_cancer_shared']['expression'], merged_hpa_by_lnc[merged_hpa_by_lnc['cluster'] == 'promote_im_cancer_specific']['expression']))


# In[112]:


plt.figure(figsize=(4, 4))
sns.barplot(x='cluster', y='expression', data=merged_hpa, order=all_clusters)
plt.xticks(rotation=90) # remember to change the name in AI
plt.yticks(rotation=90)
plt.ylabel('Log2FPKM') # significance: add ** in the figure indicating both of the p-values are less than 0.01


# ### Illumina body map

# In[133]:


lincRNA_ibm_expr = df(log_ibm.loc[op_lincRNA_ibm, :]).groupby(level=0).mean()
ibm_immune = ibm_immune = ['lymph node', 'leukocyte']

# barplot
merged_ibm_by_lnc = df(index=lincRNA_ibm_expr.index, columns=['expression', 'cluster'])

merged_ibm_by_lnc['expression'] = lincRNA_ibm_expr[ibm_immune].mean(axis=1)

for c in labels['assigned'].unique():
    if c == 'noise':
        continue  
    # get subset of average expression
    tmp_linc = list(set(labels[labels['assigned'] == c].index) & set(lincRNA_ibm_expr.index))
    
    merged_ibm_by_lnc.loc[tmp_linc, 'cluster'] = c


# In[134]:


print(stat.ttest_ind(merged_ibm_by_lnc[merged_ibm_by_lnc['cluster'] == 'inhibit_im_cancer_shared']['expression'], merged_ibm_by_lnc[merged_ibm_by_lnc['cluster'] == 'inhibit_im_cancer_specific']['expression']))


# In[135]:


print(stat.ttest_ind(merged_ibm_by_lnc[merged_ibm_by_lnc['cluster'] == 'promote_im_cancer_shared']['expression'], merged_ibm_by_lnc[merged_ibm_by_lnc['cluster'] == 'promote_im_cancer_specific']['expression']))


# In[142]:


plt.figure(figsize=(4, 4))
sns.barplot(x='cluster', y='expression', data=merged_ibm_by_lnc, order=all_clusters)
plt.xticks(rotation=90) # remember to change the name in AI
plt.yticks(rotation=90)
plt.ylabel('Log2FPKM') # significance: add ** in the figure indicating both of the p-values are less than 0.01
fig = plt.gcf()
fig.savefig('figures/vectors/F1/1d_ibm_immune.pdf', transparent=True)


# In[143]:


plt.close()


# # _Figure 2_

# ## Fig. 2a: EPIC1 expression associated with immune suppression
# ### Notes: 2019.01.13
# Add labels for selected pathways: IFN response, antigen presentation, cell proliferation

# In[162]:


EPIC1_corr_200, EPIC1_fdr_200 = df(EPIC1_corr[ctype_200]), df(EPIC1_fdr[ctype_200])
# clustered the rows and columns
g = sns.clustermap(
            EPIC1_corr_200.astype(float),
            row_cluster=True, col_cluster=True)
plt.close()
reordered_x = [EPIC1_corr_200.index[i] for i in g.dendrogram_row.reordered_ind]
reordered_y = [EPIC1_corr_200.columns[i] for i in g.dendrogram_col.reordered_ind]

# reordered by cluster
EPIC1_corr_200, EPIC1_fdr_200 = df(EPIC1_corr_200.loc[reordered_x, reordered_y]), df(EPIC1_fdr_200.loc[reordered_x, reordered_y])


# In[367]:


# bo-plot for TCGA data
boPlot(corr=EPIC1_corr_200, fdr=np.clip(-np.log10(EPIC1_fdr_200), 0, 10))


# In[164]:


g = sns.clustermap(
            merge_cl_nes.fillna(0).astype(float),
            row_cluster=False, col_cluster=True)
plt.close()
reordered_cly = [merge_cl_nes.columns[i] for i in g.dendrogram_col.reordered_ind]

merge_cl_nes, merge_cl_fdr = df(merge_cl_nes.loc[reordered_x, reordered_cly]), df(merge_cl_fdr.loc[reordered_x, reordered_cly])


# In[180]:


boPlot_gsea(corr=merge_cl_nes, fdr=np.clip(-np.log10(merge_cl_fdr.fillna(1) + 0.00001), 0, 5))


# # _Figure S1_

# ## Fig. S1a: schematic.
# ### Note: 2019.01.14
# This panel has been directly processed in AI. Currently no additional modification is needed.

# ## Fig. S1b: Feature signatures in different clusters.
# ### Note: 2019.01.14
# This part of data is still pending. The current version of draft is okay to follow though.

# In[182]:


# visualize the results: ES
merge_imSig = df(columns=CR_score.columns)
for imsig in CR_score.columns:
    merge_imSig[imsig] = pd.read_csv(
        'corr_68sig_linc/representative_signature/report/' + imsig + '/gseapy.prerank.gene_sets.report.csv',
        header=0, index_col=0, sep=',')['es']
g = sns.clustermap(merge_imSig, cmap='bwr', figsize=(16, 4), linewidth=0.1, vmax=.5, vmin=-.5, row_cluster=False)
fig = plt.gcf()
fig.savefig('figures/vectors/FS1/1b_enrichment_heatmap.pdf', transparent=True)


# In[183]:


plt.close()


# In[187]:


fingerprint = g.dendrogram_col.reordered_ind[-9:]
fingerprint = [CR_score.columns[i] for i in fingerprint]

# draw distribution
f, axes = plt.subplots(9, 1, figsize=(12, 12), sharex=True)
for i in range(len(fingerprint)):
    sns.boxplot(y=labels['assigned'], x=CR_score[fingerprint[i]],
        order=all_clusters,
        palette=color_cluster, ax=axes[i])
plt.axvline(x=0., c='k', ls=':')
plt.xlim(-30, 30)
fig = plt.gcf()
fig.savefig('figures/vectors/FS1/1b_box_feature_sig.pdf', transparent=True)


# In[188]:


plt.close()


# # _Figure S2_

# ## Fig. S2a-e: correlation with other immune cell infiltration
# ### Notes: 2019.01.14
# Format as Fig. 1c-d

# In[194]:


sns.set_palette('tab20', 18, desat=1)
f, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
for i in range(len(all_clusters)):
    clt = all_clusters[i]
    plt.xlim(-.8, .8)
    tmp_linc = labels[labels['assigned'] == clt].index
    corr_tmp = CD4T_corr.loc[tmp_linc, :]
    for col in corr_tmp.columns:
        if col in ctype_200:
            sns.kdeplot(corr_tmp[col].dropna(axis=0, how='any'), label=col, ax=axes[i], legend=False)
plt.axvline(x=0.0, ls=':', c='k')
plt.legend(loc='best', ncol=3, fontsize=8)
plt.xlabel('Correlation with CD4 T cell infiltration', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/FS2/1a_kde_CD4.pdf', transparent=True)


# In[195]:


sns.set_palette('tab20', 18, desat=1)
f, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
for i in range(len(all_clusters)):
    clt = all_clusters[i]
    plt.xlim(-.8, .8)
    tmp_linc = labels[labels['assigned'] == clt].index
    corr_tmp = Macrophage_corr.loc[tmp_linc, :]
    for col in corr_tmp.columns:
        if col in ctype_200:
            sns.kdeplot(corr_tmp[col].dropna(axis=0, how='any'), label=col, ax=axes[i], legend=False)
plt.axvline(x=0.0, ls=':', c='k')
plt.legend(loc='best', ncol=3, fontsize=8)
plt.xlabel('Correlation with macrophage infiltration', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/FS2/1b_kde_macrophage.pdf', transparent=True)


# In[196]:


sns.set_palette('tab20', 18, desat=1)
f, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
for i in range(len(all_clusters)):
    clt = all_clusters[i]
    plt.xlim(-.8, .8)
    tmp_linc = labels[labels['assigned'] == clt].index
    corr_tmp = DC_corr.loc[tmp_linc, :]
    for col in corr_tmp.columns:
        if col in ctype_200:
            sns.kdeplot(corr_tmp[col].dropna(axis=0, how='any'), label=col, ax=axes[i], legend=False)
plt.axvline(x=0.0, ls=':', c='k')
plt.legend(loc='best', ncol=3, fontsize=8)
plt.xlabel('Correlation with dendritic cell infiltration', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/FS2/1c_kde_DC.pdf', transparent=True)


# In[197]:


sns.set_palette('tab20', 18, desat=1)
f, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
for i in range(len(all_clusters)):
    clt = all_clusters[i]
    plt.xlim(-.8, .8)
    tmp_linc = labels[labels['assigned'] == clt].index
    corr_tmp = Neutrophil_corr.loc[tmp_linc, :]
    for col in corr_tmp.columns:
        if col in ctype_200:
            sns.kdeplot(corr_tmp[col].dropna(axis=0, how='any'), label=col, ax=axes[i], legend=False)
plt.axvline(x=0.0, ls=':', c='k')
plt.legend(loc='best', ncol=3, fontsize=8)
plt.xlabel('Correlation with neutrophil infiltration', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/FS2/1d_kde_neutrophil.pdf', transparent=True)


# In[198]:


sns.set_palette('tab20', 18, desat=1)
f, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
for i in range(len(all_clusters)):
    clt = all_clusters[i]
    plt.xlim(-.8, .8)
    tmp_linc = labels[labels['assigned'] == clt].index
    corr_tmp = B_cell_corr.loc[tmp_linc, :]
    for col in corr_tmp.columns:
        if col in ctype_200:
            sns.kdeplot(corr_tmp[col].dropna(axis=0, how='any'), label=col, ax=axes[i], legend=False)
plt.axvline(x=0.0, ls=':', c='k')
plt.legend(loc='best', ncol=3, fontsize=8)
plt.xlabel('Correlation with B cell infiltration', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/FS2/1e_kde_B_cell.pdf', transparent=True)


# # _Figure S3_

# ## Fig. S3a: Distribution of the beta values at promoter region
# ### Note: 2019.01.16
# Add a p-value between major clusters in each panel.

# In[206]:


methy_p200 = ['TCGA-BLCA', 'TCGA-SKCM', 'TCGA-PRAD', 'TCGA-LIHC', 'TCGA-BRCA', 'TCGA-LUSC', 'TCGA-COAD_READ', 'TCGA-SARC', 'TCGA-LUAD', 'TCGA-UCEC', 'TCGA-KIRC', 'TCGA-CESC', 'TCGA-STAD', 'TCGA-THCA', 'TCGA-HNSC', 'TCGA-KIRP']


# In[245]:


# distribution of the mean (by patients): raw tumor
f, axes = plt.subplots(4, 4, figsize=(16, 10))
old_to_new = {'inhibit_im_c3': 'inhibit_im_cancer_shared',
              'inhibit_im_c1': 'inhibit_im_cancer_specific',
              'promote_im_c0': 'promote_im_cancer_specific',
              'promote_im_c2': 'promote_im_cancer_shared'}
for ct, ax_t in zip(methy_p200, axes.flatten()):
    tmp_p = []
    tmp_i = []
    for k in old_to_new.keys():
        tmp_set = pd.read_csv(
            'CR_methylation/methy_by_cancer_type/' + ct + '/raw_methylation_probe_' + k + '.csv',
            header=0, index_col=0, sep=',').mean(axis=0)
        sns.kdeplot(tmp_set, color=color_cluster[old_to_new[k]], ax=ax_t)

        if k.split('_')[0] == 'promote':
            tmp_p.extend(tmp_set)
        else:
            tmp_i.extend(tmp_set)
    
    # set the interface to current axis
    plt.sca(ax_t)
    plt.yticks(rotation=90, fontsize=16)
    plt.xticks(fontsize=16)
    plt.title(
        ct + ', p = ' + str(np.format_float_scientific(
            stat.ks_2samp(tmp_p, tmp_i)[1], precision=2, exp_digits=3)))
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/FS3/1a_methylation.pdf', transparent=True)


# ## Fig. S3b: distribution of copy number alterations in each cluster
# ### Notes: 2019.01.16
# Similar as S3a.

# In[249]:


cnv_ctype = ['BLCA', 'CRC', 'GBM', 'LGG', 'SARC', 'LIHC', 'STAD', 'OV', 'KIRC', 'LUSC', 'KIRP', 'LUAD', 'BRCA', 'HNSC', 'UCEC', 'THCA', 'SKCM', 'PRAD', 'CESC']
cnv_ctype.sort()
print(cnv_ctype)


# In[258]:


f, axes = plt.subplots(5, 4, figsize=(14, 10), sharex=False, sharey=True)
old_to_new = {'inhibit_im_c3': 'inhibit_im_cancer_shared',
              'inhibit_im_c1': 'inhibit_im_cancer_specific',
              'promote_im_c0': 'promote_im_cancer_specific',
              'promote_im_c2': 'promote_im_cancer_shared'}

# distribution of the mean (by patient)
for ct, ax_t in zip(cnv_ctype, axes.flatten()):
    tmp_cnv = pd.read_csv(
        '/home/yuew/Documents/TCGA/copy_number/CNV_lincRNA/raw_results/' + ct + '/GDC_TCGA_patient_lincRNA_CNV.csv',
        header=0, index_col=0, sep=',')
    tmp_cnv = tmp_cnv.rename(index=gene_name['gene_symbol'])
    tmp_p = []
    if ct == 'SKCM':
        for p in tmp_cnv.columns:
            if p[13:15] == '06':
                tmp_p.append(p)
    else:
        for p in tmp_cnv.columns:
            if p[13:15] == '01':
                tmp_p.append(p)
    if len(tmp_p) < 200:
        continue
    tmp_cnv = df(tmp_cnv[tmp_p])
    tmp_p = []
    tmp_i = []
    plt.figure(figsize=(4, 2))
    for k in old_to_new.keys():
        if k =='noise':
            continue
        tmp_set = tmp_cnv.loc[labels[labels['assigned'] == old_to_new[k]].index, :].mean(axis=0)
        sns.kdeplot(tmp_set, color=color_cluster[old_to_new[k]], cumulative=True, ax=ax_t)
        tmp_set = tmp_cnv.loc[labels[labels['assigned'] == old_to_new[k]].index, :].mean(axis=0)
        if k.split('_')[0] == 'promote':
            tmp_p.extend(tmp_set)
        else:
            tmp_i.extend(tmp_set)
    # set the interface to current axis
    plt.sca(ax_t)
    plt.yticks(rotation=90, fontsize=16)
    plt.xticks(fontsize=16)
    plt.axvline(x=0., ls=':', c='k')
    plt.title(
        ct + ', p = ' + str(np.format_float_scientific(
            stat.ks_2samp(tmp_p, tmp_i)[1], precision=2, exp_digits=2)))
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/FS3/1b_cnv.pdf', transparent=True)


# # _Figure S4_

# ## Fig. S4a: CDF of overexpressed lncRNAs in tumor and normal samples
# ### Notes: 2019.01.16
# Add ks test to the panel.

# In[261]:


# calculate proportion of overexpressed in T or overexpressed in N
proportion_expr = df(columns=['Tumor_high', 'Normal_high', 'cluster_assigned'])
for ctype in wilcox.columns:
    for clt in all_clusters:
        tmp_linc = labels[labels['assigned'] == clt].index
        tmp_wil = df(wilcox.loc[tmp_linc, ctype])
        tmp_T = len(tmp_wil[tmp_wil[ctype] >= 2.].index)
        tmp_N = len(tmp_wil[tmp_wil[ctype] <= -2.].index)
        tmp_all = len(tmp_linc)
        proportion_expr.at[ctype + '|' + clt, 'Tumor_high'] = tmp_T / tmp_all
        proportion_expr.at[ctype + '|' + clt, 'Normal_high'] = tmp_N / tmp_all
        proportion_expr.at[ctype + '|' + clt, 'cluster_assigned'] = clt


# In[285]:


# high in tumor
tmp_p = []
tmp_i = []
plt.figure(figsize=(5, 4))
for clt in proportion_expr['cluster_assigned'].unique():
    tmp_c = proportion_expr[proportion_expr['cluster_assigned'] == clt]
    if clt.split('_')[0] == 'promote':
            tmp_p.extend(tmp_c['Tumor_high'])
    else:
            tmp_i.extend(tmp_c['Tumor_high'])
    sns.kdeplot(tmp_c['Tumor_high'], c=color_cluster[clt], cumulative=True)
plt.title('High expression in tumor, p = ' + str(np.format_float_scientific(stat.ks_2samp(tmp_p, tmp_i)[1], precision=2, exp_digits=2)))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16, rotation=90)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/FS4/1a_tumor.pdf', transparent=True)
plt.close()


# In[287]:


# high in normal
tmp_p = []
tmp_i = []
plt.figure(figsize=(5, 4))
for clt in proportion_expr['cluster_assigned'].unique():
    tmp_c = proportion_expr[proportion_expr['cluster_assigned'] == clt]
    if clt.split('_')[0] == 'promote':
            tmp_p.extend(tmp_c['Normal_high'])
    else:
            tmp_i.extend(tmp_c['Normal_high'])
    sns.kdeplot(tmp_c['Normal_high'], label=clt, c=color_cluster[clt], cumulative=True)
plt.title('High expression in normal, p = ' + str(np.format_float_scientific(stat.ks_2samp(tmp_p, tmp_i)[1], precision=2, exp_digits=2)))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16, rotation=90)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/FS4/1a_normal.pdf', transparent=True)
plt.close()


# ## Fig. S4b: Average p values of wilcoxon test among clusters
# ### Notes: 2019.01.16
# Keep current version.

# In[297]:


average_p_value = df(wilcox.dropna(axis=1, how='all').fillna(0).loc[labels[labels['assigned'] !='noise'].index, :].mean(1))

plt.figure(figsize=(6, 5))
sns.barplot(
    x=labels['assigned'],
    y=average_p_value[0],
    order=all_clusters,
    palette=color_cluster)
plt.ylabel('Average signed log(P-value)', fontsize=16)
plt.xlabel('')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16, rotation=90)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/FS4/1b_average_p.pdf', transparent=True)
plt.close()


# ## Fig. S4c: Expression in Human Protein Atlas

# In[316]:


f, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
for p, ax_t in zip(['spleen', 'lymph node', 'tonsil', 'bone marrow'], axes.flatten()):
    sns.boxplot(
        x=labels['assigned'], y=lincRNA_hpa_expr[p],
        order=all_clusters,
        palette=color_cluster, ax=ax_t)
    plt.sca(ax_t)
    plt.yticks(rotation=90, fontsize=16)
    plt.xticks(fontsize=12, rotation=90)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(p, fontsize=16)
fig = plt.gcf()
fig.savefig('figures/vectors/FS4/1c_HPA_immune.pdf', transparent=True)
plt.show()


# In[325]:


sns.clustermap(
    lincRNA_hpa_expr,
    row_colors=row_cluster_colors,
    vmin=-4, vmax=4,
    figsize=(8, 6),
    z_score=0, cmap='bwr')

fig = plt.gcf()
fig.savefig('figures/vectors/FS4/1c_HPA_heatmap.pdf', transparent=True)
plt.show()


# ## Fig. S4d: Expression in Illumina body map
# ### Notes: 2019.01.17
# Keep the current version in draft.

# In[328]:


f, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
for p, ax_t in zip(['lymph node', 'leukocyte'], axes.flatten()):
    sns.boxplot(
        x=labels['assigned'], y=lincRNA_ibm_expr[p],
        order=all_clusters,
        palette=color_cluster, ax=ax_t)
    plt.sca(ax_t)
    plt.yticks(rotation=90, fontsize=16)
    plt.xticks(fontsize=12, rotation=90)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(p, fontsize=16)
fig = plt.gcf()
fig.savefig('figures/vectors/FS4/1d_IBM_immune.pdf', transparent=True)
plt.show()


# In[329]:


sns.clustermap(
    lincRNA_ibm_expr,
    row_colors=row_cluster_colors,
    vmin=-4, vmax=4,
    figsize=(8, 6),
    z_score=0, cmap='bwr')

fig = plt.gcf()
fig.savefig('figures/vectors/FS4/1d_IBM_heatmap.pdf', transparent=True)
plt.show()


# # _Figure S5_

# ## Fig. S5a: MHC-I and proliferation score on tSNE map
# ### Notes: 2019.01.17
# Keep the current version in the draft

# In[331]:


# exclude the noise
coordinates_signal = df(coordinates.loc[labels[labels['cluster'] != -1].index, :])

# add color as MHC.I
plt.figure(figsize=(6, 6))
sns.scatterplot(
    x=coordinates_signal['tSNE-1'], y=coordinates_signal['tSNE-2'],
    hue=np.sign(CR_score['MHC1_21978456']), style=labels['assigned'],
    size=abs(CR_score['MHC1_21978456']),
    markers=marker_cluster,
    linewidth=0.01,
    alpha=.75, palette={-1: 'blue', 1: 'red', 0: 'white'})
plt.legend(prop={'size': 12})
plt.xlabel('TSNE-1', fontsize=18)
plt.ylabel('TSNE-2', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18, rotation=90)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/FS5/tSNE_MHC1_cluster.pdf', transparent=True)


# In[332]:


# exclude the noise
coordinates_signal = df(coordinates.loc[labels[labels['cluster'] != -1].index, :])

# add color as MHC.I
plt.figure(figsize=(6, 6))
sns.scatterplot(
    x=coordinates_signal['tSNE-1'], y=coordinates_signal['tSNE-2'],
    hue=np.sign(CR_score['Module11_Prolif_score']), style=labels['assigned'],
    size=abs(CR_score['Module11_Prolif_score']),
    markers=marker_cluster,
    linewidth=0.01,
    alpha=.75, palette={-1: 'blue', 1: 'red', 0: 'white'})
plt.legend(prop={'size': 12})
plt.xlabel('TSNE-1', fontsize=18)
plt.ylabel('TSNE-2', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18, rotation=90)
plt.tight_layout()
fig = plt.gcf()
fig.savefig('figures/vectors/FS5/tSNE_proliferation_cluster.pdf', transparent=True)


# ## Fig. S5b: Candidate lincRNAs for tumor intrinsic immune evasion
# ### Notes: 2019.01.17
# Keep the current version in the draft

# In[334]:


# EPIC1 in cluster 1
cluster_1 = CR_score.loc[labels[labels['cluster'] == 1].index, :]

# sort by module 11 and MHCI: regulate 25% of the cancer types (8)
mhc1_1 = df(cluster_1[cluster_1['MHC1_21978456'] <= -8])
mhc1_1 = df(mhc1_1[mhc1_1['MHC.I_19272155'] <= -8])
prolif_1 = df(cluster_1[cluster_1['Module11_Prolif_score'] >= 8])
op_mhc1_prolif_1 = list(set(mhc1_1.index) & set(prolif_1.index))
print(len(op_mhc1_prolif_1))


# In[336]:


df(wilcox.loc[op_mhc1_prolif_1, :].mean(axis=1)).sort_values(by=0, ascending=False)


# In[344]:


ranked_list = df(wilcox.loc[op_mhc1_prolif_1, :].mean(axis=1)).sort_values(by=0, ascending=False)
plt.figure(figsize=(2, 8))
sns.barplot(x=ranked_list[0], y=ranked_list.index, color='gray')
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.xlabel('Average p value (tumor versus normal)', fontsize=18)
fig = plt.gcf()
fig.savefig('figures/vectors/FS5/5b_candidates_bar.pdf', transparent=True)


# In[345]:


sns.clustermap(CR_score.loc[ranked_list.index, :], cmap='bwr', row_cluster=False, figsize=(16, 4), vmin=-8, vmax=8)
fig = plt.gcf()
fig.savefig('figures/vectors/FS5/5b_candidates_heatmap.pdf', transparent=True)


# ## Fig. S5c: Gene-level correlation between EPIC1 and MHCI/proliferation genes
# ### Notes: 2019.01.17
# Keep current version in the draft.

# In[346]:


mhc1 = ['HLA-G', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-E', 'HLA-F', 'HLA-J']
prolif = ['CDKN3', 'NDC80', 'RNASEH2A', 'CENPA', 'SMC2', 'CENPE', 'RAD51AP1', 'PLK4', 'NMU',
          'KIF2C', 'TMSB15A', 'UBE2C', 'CHEK1', 'ZWINT', 'OIP5', 'CRABP1', 'ECT2', 'EIF4EBP1', 'EZH2', 'FEN1', 'HSPA4L',
          'TPX2', 'FOXM1', 'NCAPH', 'PRAME', 'PDSS1', 'KIF4A', 'RAD54B', 'ASPM', 'FBXO5', 'ATAD2', 'RACGAP1', 'GPSM2',
          'DONSON', 'HMMR', 'BIRC5', 'KIF11', 'LMNB1', 'MAD2L1', 'MCM4', 'MCM5', 'MKI67', 'MMP1', 'MYBL1', 'MYBL2', 'NEK2', 'NUSAP1', 'GTSE1', 'GINS2', 'PLK1', 'FAM64A', 'ERCC6L',
          'NCAPG2', 'CEP55', 'FANCI', 'HJURP', 'MCM10', 'DEPDC1', 'C1orf112', 'CENPN', 'PBK', 'KIF15', 'CIAPIN1', 'ACTR3B',
          'GPR126', 'SPC25', 'RAD21', 'RFC3', 'RFC4', 'RRM2', 'NCAPG', 'STIL', 'SKP2', 'SOX11', 'SQLE', 'AURKA', 'TAF2', 'TARS',
          'BUB1B', 'TK1', 'TMPO', 'TOP2A', 'PHLDA2', 'TTK', 'LRP8', 'DSCC1', 'MLF1IP', 'E2F8', 'SHCBP1', 'SLC7A5', 'ANP32E',
          'KIF18A', 'CDC7', 'CDC45', 'RAD54L', 'TTF2', 'PIR', 'ACTL6A', 'GGH', 'CCNA2', 'CCNB1', 'PRC1', 'CCNB2', 'CCNE2', 'EXO1',
          'AURKB', 'PTTG1', 'TRIP13', 'KIF23', 'APOBEC3B', 'MTFR1', 'ESPL1', 'DLGAP5', 'CDK1', 'MELK', 'GINS1', 'CDC6', 'CDC20', 'NCAPD2', 'KIF14']


# In[349]:


# selected pathways: MHCI and proliferation
select_genes = ['HLA-G', 'HLA-A', 'HLA-B', 'HLA-C', 'HLA-E',
                'HLA-F', 'HLA-J', 'CDKN3', 'NDC80', 'RNASEH2A',
                'CENPA', 'SMC2', 'CENPE', 'RAD51AP1', 'PLK4', 'NMU',
                'KIF2C', 'TMSB15A', 'UBE2C', 'CHEK1', 'ZWINT', 'OIP5',
                'CRABP1', 'ECT2', 'EIF4EBP1', 'EZH2', 'FEN1', 'HSPA4L',
                'TPX2', 'FOXM1', 'NCAPH', 'PRAME', 'PDSS1', 'KIF4A',
                'RAD54B', 'ASPM', 'FBXO5', 'ATAD2', 'RACGAP1', 'GPSM2',
                'DONSON', 'HMMR', 'BIRC5', 'KIF11', 'LMNB1', 'MAD2L1',
                'MCM4', 'MCM5', 'MKI67', 'MMP1', 'MYBL1', 'MYBL2', 'NEK2',
                'NUSAP1', 'GTSE1', 'GINS2', 'PLK1', 'FAM64A', 'ERCC6L',
                'NCAPG2', 'CEP55', 'FANCI', 'HJURP', 'MCM10', 'DEPDC1',
                'C1orf112', 'CENPN', 'PBK', 'KIF15', 'CIAPIN1', 'ACTR3B',
                'GPR126', 'SPC25', 'RAD21', 'RFC3', 'RFC4', 'RRM2', 'NCAPG',
                'STIL', 'SKP2', 'SOX11', 'SQLE', 'AURKA', 'TAF2', 'TARS',
                'BUB1B', 'TK1', 'TMPO', 'TOP2A', 'PHLDA2', 'TTK', 'LRP8',
                'DSCC1', 'MLF1IP', 'E2F8', 'SHCBP1', 'SLC7A5', 'ANP32E',
                'KIF18A', 'CDC7', 'CDC45', 'RAD54L', 'TTF2', 'PIR', 'ACTL6A',
                'GGH', 'CCNA2', 'CCNB1', 'PRC1', 'CCNB2', 'CCNE2', 'EXO1',
                'AURKB', 'PTTG1', 'TRIP13', 'KIF23', 'APOBEC3B', 'MTFR1',
                'ESPL1', 'DLGAP5', 'CDK1', 'MELK', 'GINS1', 'CDC6', 'CDC20', 'NCAPD2', 'KIF14']


# In[364]:


# correlation
corr_cand_selected = {}

for ctype in ctype_200:
    corr_cand_selected[ctype] = {'corr': [], 'p': [], 'fdr_bool': []}
    corr_cand_selected[ctype]['corr'], corr_cand_selected[ctype]['p'], corr_cand_selected[ctype]['fdr_bool'] = corr_pcg_lnc(
        pcg_list=select_genes,
        lnc_list=['EPIC1'],
        ctype=ctype,
        convert_id=gene_name)
    print(ctype + ' finished')


# In[388]:


# EPIC1 association with genes
corrmatrix_EPIC1 = df(columns=ctype_200, index=select_genes)
pmatrix_EPIC1 = df(columns=ctype_200, index=select_genes)
spmatrix_EPIC1 = df(columns=ctype_200, index=select_genes)
for ctype in ctype_200:
    tmp_corr = df(corr_cand_selected[ctype]['corr']['EPIC1'])
    tmp_p = df(corr_cand_selected[ctype]['p']['EPIC1'])
    corrmatrix_EPIC1[ctype] = tmp_corr['EPIC1'].astype(float)
    pmatrix_EPIC1[ctype] = -np.log10(tmp_p['EPIC1'].astype(float))
    spmatrix_EPIC1[ctype] = -np.log10(tmp_p['EPIC1'].astype(float)) * np.sign(tmp_corr['EPIC1'].astype(float))


# In[390]:


sns.clustermap(
    spmatrix_EPIC1.dropna(axis=0, how='all').dropna(axis=1, how='all').fillna(0).T,
    col_cluster=True,
    figsize=(30, 6),
    cmap='bwr', vmin=-5, vmax=5)
fig = plt.gcf()
fig.savefig('figures/vectors/FS5/5c_gene_corr_heatmap.pdf', transparent=True)


# In[394]:


sns.clustermap(
    spmatrix_EPIC1.loc[mhc1, :].dropna(axis=0, how='all').dropna(axis=1, how='all').fillna(0).T,
    col_cluster=True,
    figsize=(6, 6),
    cmap='bwr', vmin=-4, vmax=4)
fig = plt.gcf()
fig.savefig('figures/vectors/FS5/5c_mhc1_gene_corr_heatmap.pdf', transparent=True)


# In[397]:


sns.clustermap(
    spmatrix_EPIC1.loc[prolif, :].dropna(axis=0, how='all').dropna(axis=1, how='all').fillna(0).T,
    col_cluster=True,
    figsize=(10, 6),
    cmap='bwr', vmin=-4, vmax=4)
fig = plt.gcf()
fig.savefig('figures/vectors/FS5/5c_proliferation_gene_corr_heatmap.pdf', transparent=True)

