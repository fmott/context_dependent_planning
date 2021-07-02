"""
Florian Ott, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob as glob 
import pandas as pd 
import time as time 
import scipy.stats as stats
import matplotlib.patches as mpatches
plt.style.use('./ara.mplstyle')

#%% Load behavioural data 
filename = glob.glob('../data/behaviour/data_all_partipants_20210623102746.csv')
dat = pd.read_csv(filename[0],index_col = 0)

#%% Load model data
# Posterior mean and credibility intervals of estimated parameters
with open("../data/model/summary_hybrid_20210621151244.pkl", "rb") as f:
    summary_hybrid = pickle.load(f)['summary']
with open("../data/model/summary_RT_vs_conflict_20210621151244.pkl", "rb") as f:
    summary_RTConf = pickle.load(f)['summary']    

# Posterior samples
filename = glob.glob('../data/model/posterior_samples_RT_vs_conflict_20210702153146.csv') 
samples_RTConf  = pd.read_csv(filename[0],index_col = 0)
#%% Generate Figure 3 
fig = plt.figure(constrained_layout=True,figsize=[7,2])
gs = fig.add_gridspec(1, 24)
ax0 = fig.add_subplot(gs[0:7])
ax1 = fig.add_subplot(gs[7:14])
ax2 = fig.add_subplot(gs[14:24])

##############################################################################
# Subplot 1 and 2 - Post Hoc correlations 
##############################################################################
# Get posterior mean of partipant level planning weights
y = summary_hybrid['beta_dv']['mean'].to_numpy().squeeze()

# Get total accumulated points for each participant
idx = dat['index'] == 239
points = dat.loc[idx,['points_after']].to_numpy().squeeze() 
rt = dat.groupby('vpn')['reaction_time'].mean().to_numpy()

# Calculate correlation and regression line 
pars1 = np.polyfit(points,y, 1)
xx1 = np.arange(np.min(points),np.max(points),0.1)
yy1 = pars1[0]*xx1 + pars1[1]
r1,p1 = stats.pearsonr(points,y)

pars2 = np.polyfit(rt,y, 1)
xx2 = np.arange(np.min(rt),np.max(rt),0.1)
yy2 = pars2[0]*xx2 + pars2[1]
r2,p2 = stats.pearsonr(rt,y)

# Plotting
ax0.scatter(points,y,color='black',edgecolor='black',facecolor='white',alpha=0.6,zorder=10,s=20,marker='.',linewidth=1)
ax0.plot(xx1,yy1,color='black',linewidth=1)
ax0.set_ylabel(r'$\beta_{plan}$',fontsize = 12)
ax0.set_xlabel('Points')
ax0.text(-0.1,1.1,'A',transform=ax0.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')

ax1.scatter(rt,y,color='black',edgecolor='black',facecolor='white',alpha=0.6,zorder=10,s=20,marker='.',linewidth=1)
ax1.plot(xx2,yy2,color='black',linewidth=1)
ax1.set_ylabel(r'$\beta_{plan}$',fontsize = 12)
ax1.set_xlabel('Response time [s]')
ax1.set_xticks([0.5,0.75,1,1.25,1.5])
ax1.text(-0.1,1.1,'B',transform=ax1.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')


##############################################################################
# Subplot 3 - Reaction time and choice conflict  
##############################################################################
# Compute population regression lines for intermediate and extreme offers  

# Get indices for intermedaite and extreme trials
idx_14 = (dat['timeout']==0) & ((dat['reward']==1) | (dat['reward']==4))
idx_23 = (dat['timeout']==0) & ((dat['reward']==2) | (dat['reward']==3))

# Get minimum and maximum conflict for the intermediate and extreme context
# and compute predicted respone times over this range. Note that we 
# exponetiate the linear predictor since we fitted RTs on the log scale.
max_14 = np.max(dat.loc[idx_14,['conflict']]).to_numpy().squeeze()
min_14 = np.min(dat.loc[idx_14,['conflict']]).to_numpy().squeeze()
max_23 = np.max(dat.loc[idx_23,['conflict']]).to_numpy().squeeze()
min_23 = np.min(dat.loc[idx_23,['conflict']]).to_numpy().squeeze()

xx_14 = np.expand_dims(np.arange(min_14,max_14,0.01),1)
xx_23 = np.expand_dims(np.arange(min_23,max_23,0.01),1)

ypred_mus_easy = np.exp(samples_RTConf['mu_intercept'].to_numpy() + samples_RTConf['mu_beta_conflict'].to_numpy()*xx_14 + samples_RTConf['mu_beta_ishard'].to_numpy() * 0 + samples_RTConf['mu_beta_interaction'].to_numpy()*xx_14*0)
ypred_mus_hard = np.exp(samples_RTConf['mu_intercept'].to_numpy() + samples_RTConf['mu_beta_conflict'].to_numpy()*xx_23 + samples_RTConf['mu_beta_ishard'].to_numpy() * 1 + samples_RTConf['mu_beta_interaction'].to_numpy()*xx_23*1)

# We have one predicted response time for each conflict level and posterior
# sample. Now, compute percentiles across posterior samples.  
ypred_mus_easy_median = np.percentile(ypred_mus_easy,50,axis=1)
ypred_mus_easy_p025 = np.percentile(ypred_mus_easy,2.5,axis=1)
ypred_mus_easy_p975 = np.percentile(ypred_mus_easy,97.5,axis=1)

ypred_mus_hard_median = np.percentile(ypred_mus_hard,50,axis=1)
ypred_mus_hard_p025 = np.percentile(ypred_mus_hard,2.5,axis=1)
ypred_mus_hard_p975 = np.percentile(ypred_mus_hard,97.5,axis=1)

# Plotting
patch1 = mpatches.Patch(color='darkorange',label='Offer 1 and 4')
patch2 = mpatches.Patch(color='navy',label='Offer 2 and 3')

ax2.plot(xx_14.squeeze(),ypred_mus_easy_median,color='darkorange');
ax2.fill_between(xx_14.squeeze(),ypred_mus_easy_p025,ypred_mus_easy_p975,facecolor='darkorange',alpha=0.3);
ax2.plot(xx_23.squeeze(),ypred_mus_hard_median,color='navy');
ax2.fill_between(xx_23.squeeze(),ypred_mus_hard_p025,ypred_mus_hard_p975,facecolor='navy',alpha=0.2);
ax2.set_ylabel('Response time [s]');
ax2.set_xlabel('Conflict');
ax2.legend(handles=[patch1, patch2],loc='upper left',ncol=1,frameon=False)
ax2.text(-0.1,1.1,'C',transform=ax2.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')

plt.tight_layout()
timestr = time.strftime("%Y%m%d%H%M%S")
# fig.savefig('fig3_' + timestr + '.tif', dpi=300, bbox_inches='tight', transparent=False)
# fig.savefig('fig3_' + timestr + '.png', dpi=300, bbox_inches='tight', transparent=False)
