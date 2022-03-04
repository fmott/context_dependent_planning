"""
Florian Ott, 2022
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
filename = glob.glob('../data/behaviour/data_all_participants_20220215120148.csv')
dat = pd.read_csv(filename[0],index_col = 0)

unique_vpns = np.unique(dat['vpn'])
ns = len(unique_vpns)
#%% Load model data
# Posterior mean and credibility intervals of estimated parameters
with open("../data/model/summary_RT_vs_conflict_20220218115905.pkl", "rb") as f:
    summary_RTConf = pickle.load(f)['summary']    

# Posterior samples
with open('../data/model/samples_RT_vs_conflict_20220218115905.pkl', "rb") as f:
    samples_RTConf  = pickle.load(f)['posterior_samples']    
#%% Generate Figure 3 
fig = plt.figure(constrained_layout=True,figsize=[4.5,2])
gs = fig.add_gridspec(1, 24)
ax0 = fig.add_subplot(gs[0:10])
ax1 = fig.add_subplot(gs[10:24])

##############################################################################
# Subplot 1 - Plot Parameters
##############################################################################
param_names_tmp = ['mu_beta_conflict','mu_beta_ishard','mu_beta_interaction']
plvl_param_names = ['beta_conflict','beta_ishard','beta_interaction']
yticklabels = [r'$\beta_{conflict}$',r'$\beta_{intermediate}$',r'$\beta_{interaction}$']

for i, param in enumerate(param_names_tmp):   
    # Plot PDF
    kernel = stats.gaussian_kde(samples_RTConf[param])
    min_tmp = np.mean(samples_RTConf[param])+np.std(samples_RTConf[param])*4
    max_tmp = np.mean(samples_RTConf[param])-np.std(samples_RTConf[param])*4
    positions = np.linspace(start=min_tmp,stop=max_tmp,num=50)
    pdf_estimate = kernel(positions)
    pdf_estimate = pdf_estimate/ (np.max(pdf_estimate) + 0.3) # scale
    ax0.plot(positions,pdf_estimate + i,color='grey',linewidth=1)
    ax0.fill_between(positions,np.ones(len(positions)) * i, pdf_estimate + i,alpha=0.3,color = 'grey')

    # Plot posterior mean and 95% credibility intervals
    ax0.scatter(summary_RTConf[param]['mean'].to_numpy(),i,color='black',s=10)
    ax0.hlines(i, xmin=summary_RTConf[param]['q2_5'].to_numpy(), xmax=summary_RTConf[param]['q97_5'].to_numpy(),color='black',linewidth=1)

    # Plot posterior mean of participant level estimates 
    jitter = np.random.normal(loc=-0.16,scale=0.05,size=ns)
    ax0.scatter(summary_RTConf[plvl_param_names[i]]['mean'],i+jitter,edgecolor='black',facecolor='white',alpha=0.3,s=10,marker='.',linewidth=1,zorder=0)
        
ax0.vlines(x= 0, ymin=-0.5,ymax= 3,color='black',linestyle=':',zorder=0)

ax0.set_yticks(range(len(param_names_tmp)))
ax0.set_yticklabels(yticklabels,rotation=0,y=np.arange(2,len(param_names_tmp)+1),fontsize = 10)
ax0.set_ylim((-0.5,len(param_names_tmp)))
ax0.text(-0.1,1.1,'A',transform=ax0.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')
ax0.set_xticks([0,0.1,0.2])

##############################################################################
# Subplot 2 - Reaction time and choice conflict  
##############################################################################
# Compute population regression lines for intermediate and extreme offers  

# Get indices for intermedaite and extreme trials
idx_14 = (dat['timeout']==0) & ((dat['reward']==1) | (dat['reward']==4))
idx_23 = (dat['timeout']==0) & ((dat['reward']==2) | (dat['reward']==3))

# Get minimum and maximum conflict for the intermediate and extreme context
# and compute predicted respone times over this range. Note that we 
# exponetiate the linear predictor since we fitted RTs on the log scale.
max_14 = np.max(dat.loc[idx_14,['conflict_planning']]).to_numpy().squeeze()
min_14 = np.min(dat.loc[idx_14,['conflict_planning']]).to_numpy().squeeze()
max_23 = np.max(dat.loc[idx_23,['conflict_planning']]).to_numpy().squeeze()
min_23 = np.min(dat.loc[idx_23,['conflict_planning']]).to_numpy().squeeze()

xx_14 = np.expand_dims(np.arange(min_14,max_14,0.01),1)
xx_23 = np.expand_dims(np.arange(min_23,max_23,0.01),1)

ypred_mus_easy = np.exp(samples_RTConf['mu_intercept'] + samples_RTConf['mu_beta_conflict']*xx_14 + samples_RTConf['mu_beta_ishard'] * 0 + samples_RTConf['mu_beta_interaction']*xx_14*0)
ypred_mus_hard = np.exp(samples_RTConf['mu_intercept'] + samples_RTConf['mu_beta_conflict']*xx_23 + samples_RTConf['mu_beta_ishard'] * 1 + samples_RTConf['mu_beta_interaction']*xx_23*1)

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

ax1.plot(xx_14.squeeze(),ypred_mus_easy_median,color='darkorange');
ax1.fill_between(xx_14.squeeze(),ypred_mus_easy_p025,ypred_mus_easy_p975,facecolor='darkorange',alpha=0.3);
ax1.plot(xx_23.squeeze(),ypred_mus_hard_median,color='navy');
ax1.fill_between(xx_23.squeeze(),ypred_mus_hard_p025,ypred_mus_hard_p975,facecolor='navy',alpha=0.2);
ax1.set_ylabel('Response time [s]');
ax1.set_xlabel('Conflict');
ax1.legend(handles=[patch1, patch2],loc='upper left',ncol=1,frameon=False)
ax1.text(-0.1,1.1,'B',transform=ax1.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')
plt.tight_layout()
timestr = time.strftime("%Y%m%d%H%M%S")
# fig.savefig('fig4_' + timestr + '.tif', dpi=1000, bbox_inches='tight', transparent=False)