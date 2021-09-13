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
import matplotlib.gridspec as gsp
plt.style.use('./ara.mplstyle')

#%% Load behavioural data 
filename = glob.glob('../data/behaviour/data_all_partipants_20210623102746.csv')
dat = pd.read_csv(filename[0],index_col = 0)

#%% Load model data
# Posterior mean and credibility intervals of estimated parameters
with open("../data/model/summary_hybrid_20210621151244.pkl", "rb") as f:
    summary_hybrid = pickle.load(f)['summary']

# Posterior samples
filename = glob.glob('../data/model/posterior_samples_hybrid_20210702153146.csv') 
samples_hybrid = pd.read_csv(filename[0],index_col = 0)

# Posterior predictive simulations 
filename = glob.glob('../data/model/ppc_hybrid_20210702153146.csv') 
ppc_hybrid = pd.read_csv(filename[0],index_col = None, header = None).to_numpy() == 0

filename = glob.glob('../data/model/ppc_simple_20210906104206.csv') 
ppc_simple = pd.read_csv(filename[0],index_col = None, header = None).to_numpy() == 0

filename = glob.glob('../data/model/ppc_planning_20210906104206.csv') 
ppc_planning = pd.read_csv(filename[0],index_col = None, header = None).to_numpy() == 0

# LOOIC for model comparison plot
filename = glob.glob('../data/model/model_comparison.csv') 
looic = pd.read_csv(filename[0],index_col = 0)

#%% Calculate choice frequencies
# Create subset of the data like it was used during fitting
idx = (dat['timeout'] == 0)
dat_subset = dat.loc[idx]

# List of participant IDs
unique_vpn = np.unique(dat['vpn'])
ns = len(unique_vpn)

# Calculate response frequencies across offer values for participants and
# posterior predictive simulations
# Hybrid
n_sample = 100
p_accept_ov_ppc_hm = np.zeros((ns,n_sample,4,2))

for s,vpn in enumerate(unique_vpn):
    for sim in range(n_sample):
        for ov in range(4):
            idx1 = (dat_subset['reward'] == ov + 1) & (ppc_hybrid[sim,:] == 0) & (dat_subset['vpn'] == vpn) & (dat_subset['is_basic'] == 1)
            idx2 = (dat_subset['reward'] == ov + 1) & (ppc_hybrid[sim,:] == 1) & (dat_subset['vpn'] == vpn) & (dat_subset['is_basic'] == 1)
            p_accept_ov_ppc_hm[s,sim,ov,0] = np.sum(idx1) / ( np.sum(idx1) + np.sum(idx2) )
            p_accept_ov_ppc_hm[s,sim,ov,1] = np.sum(idx2) / ( np.sum(idx1) + np.sum(idx2) )

p_accept_ov_ppc_m_hm = np.nanmean(p_accept_ov_ppc_hm,1) # mean over posterior samples
p_accept_ov_ppc_s_hm = np.nanstd(p_accept_ov_ppc_hm,1)
p_accept_ov_ppc_m_m_hm = np.nanmean(p_accept_ov_ppc_m_hm,0) # mean over simulated participants
p_accept_ov_ppc_s_s_hm = np.nanstd(p_accept_ov_ppc_m_hm,0)

# Simple model
n_sample = 100
p_accept_ov_ppc_sm = np.zeros((ns,n_sample,4,2))

for s,vpn in enumerate(unique_vpn):
    for sim in range(n_sample):
        for ov in range(4):
            idx1 = (dat_subset['reward'] == ov + 1) & (ppc_simple[sim,:] == 0) & (dat_subset['vpn'] == vpn) & (dat_subset['is_basic'] == 1)
            idx2 = (dat_subset['reward'] == ov + 1) & (ppc_simple[sim,:] == 1) & (dat_subset['vpn'] == vpn) & (dat_subset['is_basic'] == 1)
            p_accept_ov_ppc_sm[s,sim,ov,0] = np.sum(idx1) / ( np.sum(idx1) + np.sum(idx2) )
            p_accept_ov_ppc_sm[s,sim,ov,1] = np.sum(idx2) / ( np.sum(idx1) + np.sum(idx2) )

p_accept_ov_ppc_m_sm = np.nanmean(p_accept_ov_ppc_sm,1) # mean over posterior samples
p_accept_ov_ppc_s_sm = np.nanstd(p_accept_ov_ppc_sm,1)
p_accept_ov_ppc_m_m_sm = np.nanmean(p_accept_ov_ppc_m_sm,0) # mean over simulated participants
p_accept_ov_ppc_s_s_sm = np.nanstd(p_accept_ov_ppc_m_sm,0)

# Planning model
n_sample = 100
p_accept_ov_ppc_pm = np.zeros((ns,n_sample,4,2))

for s,vpn in enumerate(unique_vpn):
    for sim in range(n_sample):
        for ov in range(4):
            idx1 = (dat_subset['reward'] == ov + 1) & (ppc_planning[sim,:] == 0) & (dat_subset['vpn'] == vpn) & (dat_subset['is_basic'] == 1)
            idx2 = (dat_subset['reward'] == ov + 1) & (ppc_planning[sim,:] == 1) & (dat_subset['vpn'] == vpn) & (dat_subset['is_basic'] == 1)
            p_accept_ov_ppc_pm[s,sim,ov,0] = np.sum(idx1) / ( np.sum(idx1) + np.sum(idx2) )
            p_accept_ov_ppc_pm[s,sim,ov,1] = np.sum(idx2) / ( np.sum(idx1) + np.sum(idx2) )

p_accept_ov_ppc_m_pm = np.nanmean(p_accept_ov_ppc_pm,1) # mean over posterior samples
p_accept_ov_ppc_s_pm = np.nanstd(p_accept_ov_ppc_pm,1)
p_accept_ov_ppc_m_m_pm = np.nanmean(p_accept_ov_ppc_m_pm,0) # mean over simulated participants
p_accept_ov_ppc_s_s_pm = np.nanstd(p_accept_ov_ppc_m_pm,0)

# Participants
# Calculate response frequencies across offer values
p_accept_ov = np.zeros((ns,4,2))
p_accept_dist50 = np.zeros((ns,4)) 
for s,vpn in enumerate(unique_vpn):
    for ov in range(4):
        idx1 = (dat['reward'] == ov + 1) & (dat['response'] == 0) & (dat['vpn'] == vpn) & (dat['timeout'] == 0) & (dat['is_basic'] == 1)
        idx2 = (dat['reward'] == ov + 1) & (dat['response'] == 1) & (dat['vpn'] == vpn) & (dat['timeout'] == 0) & (dat['is_basic'] == 1)
        p_accept_ov[s,ov,0] = np.sum(idx1) / ( np.sum(idx1) + np.sum(idx2) )
        p_accept_ov[s,ov,1] = np.sum(idx2) / ( np.sum(idx1) + np.sum(idx2) )
        p_accept_dist50[s,ov] = np.absolute((np.sum(idx1) / ( np.sum(idx1) + np.sum(idx2))) - 0.5)


p_accept_ov_m = np.nanmean(p_accept_ov,0)
p_accept_ov_s = np.nanstd(p_accept_ov,0)

#%% Compute Wilcoxon signed-rank tests. Distance of choice frequencies 
# from 50%, compared between offer values.
w = np.zeros(6)
p = np.zeros(6)
comparisons = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
for i,comp in enumerate(comparisons):
    w[i],p[i] = stats.wilcoxon(p_accept_dist50[:,comp[0]],p_accept_dist50[:,comp[1]])
    
#%% Calculate matches between simulated and participant choice
match_hm_ov = np.zeros((4, len(ppc_hybrid)))
for i in range(len(ppc_hybrid)):
    for ov in range(4):
        idx = (dat_subset['reward'] == (ov+1))
        match_hm_ov[ov,i] = np.sum(ppc_hybrid[i,idx] == dat_subset.loc[idx,['response']].to_numpy().squeeze()) / np.sum(idx)
match_hm_ov_m =  np.mean(match_hm_ov,axis=1) 
match_hm_ov_s =  np.std(match_hm_ov,axis=1)  

match_sm_ov = np.zeros((4, len(ppc_simple)))
for i in range(len(ppc_hybrid)):
    for ov in range(4):
        idx = (dat_subset['reward'] == (ov+1))
        match_sm_ov[ov,i] = np.sum(ppc_simple[i,idx] == dat_subset.loc[idx,['response']].to_numpy().squeeze())/ np.sum(idx)
match_sm_ov_m =  np.mean(match_sm_ov,axis=1) 
match_sm_ov_s =  np.std(match_sm_ov,axis=1)  

match_pm_ov = np.zeros((4, len(ppc_planning)))
for i in range(len(ppc_hybrid)):
    for ov in range(4):
        idx = (dat_subset['reward'] == (ov+1))
        match_pm_ov[ov,i] = np.sum(ppc_planning[i,idx] == dat_subset.loc[idx,['response']].to_numpy().squeeze()) / np.sum(idx)
match_pm_ov_m =  np.mean(match_pm_ov,axis=1) 
match_pm_ov_s =  np.std(match_pm_ov,axis=1)  
    
#%% Generate Figure 2 
fig = plt.figure(constrained_layout=True,figsize=[4.5,4.5])
gs = fig.add_gridspec(2,15)
ax0 = fig.add_subplot(gs[0,0:11])
ax1 = fig.add_subplot(gs[0,11:15])
ax2 = fig.add_subplot(gs[1,0:6])
ax3 = fig.add_subplot(gs[1,6:15])
##############################################################################
# Subplot 1 - Choice behaviour 
##############################################################################
np.random.seed(1)
patch1 = mpatches.Patch(facecolor='white', label='Participants',edgecolor='black')
patch2 = mpatches.Patch(facecolor=[0.7,0.7,0.7], label='PM',edgecolor='black',hatch='////')
patch3 = mpatches.Patch(facecolor=[0.7,0.7,0.7], label='SM',edgecolor='black',hatch='....')
patch4 = mpatches.Patch(facecolor=[0.7,0.7,0.7], label='HM',edgecolor='black',hatch='xxxx')

jitter = np.random.normal(loc= -0.2,scale=0.1,size=ns)
barx_participant = [0,5,10,15]
barx_pm = [0+1,5+1,10+1,15+1]
barx_sm = [0+2,5+2,10+2,15+2]
barx_hm = [0+3,5+3,10+3,15+3]

xtick = [1.5,6.5,11.5,16.5]
vlines_max_part = p_accept_ov_m[:,0]+ p_accept_ov_s[:,0]
vlines_min_part = p_accept_ov_m[:,0]- p_accept_ov_s[:,0]
vlines_max_hm= p_accept_ov_ppc_m_m_hm[:,0]+ p_accept_ov_ppc_s_s_hm[:,0]
vlines_min_hm = p_accept_ov_ppc_m_m_hm[:,0]- p_accept_ov_ppc_s_s_hm[:,0]

vlines_max_pm= p_accept_ov_ppc_m_m_pm[:,0]+ p_accept_ov_ppc_s_s_pm[:,0]
vlines_min_pm = p_accept_ov_ppc_m_m_pm[:,0]- p_accept_ov_ppc_s_s_pm[:,0]

vlines_max_sm= p_accept_ov_ppc_m_m_sm[:,0]+ p_accept_ov_ppc_s_s_sm[:,0]
vlines_min_sm = p_accept_ov_ppc_m_m_sm[:,0]- p_accept_ov_ppc_s_s_sm[:,0]

ax0.hlines(0.5,xmin=-1,xmax= 19, color='black', linestyle=':',zorder=0)
ax0.vlines(x=barx_participant,ymin=vlines_min_part,ymax= vlines_max_part, color='black', linestyle='-',zorder=20)
ax0.vlines(x=barx_hm,ymin=vlines_min_hm,ymax= vlines_max_hm, color='black', linestyle='-',zorder=20)
ax0.vlines(x=barx_sm,ymin=vlines_min_sm,ymax= vlines_max_sm, color='black', linestyle='-',zorder=20)
ax0.vlines(x=barx_pm,ymin=vlines_min_pm,ymax= vlines_max_pm, color='black', linestyle='-',zorder=20)

plt.rcParams['hatch.linewidth'] = 0.5

ax0.bar(barx_participant,p_accept_ov_m[:,0],edgecolor = 'black',width=1,color='white') 
ax0.bar(barx_hm,p_accept_ov_ppc_m_m_hm[:,0],color=[0.7,0.7,0.7],edgecolor = 'black',width=1,hatch='xxxx') 
ax0.bar(barx_sm,p_accept_ov_ppc_m_m_sm[:,0],color=[0.7,0.7,0.7],edgecolor = 'black',width=1,hatch='....') 
ax0.bar(barx_pm,p_accept_ov_ppc_m_m_pm[:,0],color=[0.7,0.7,0.7],edgecolor = 'black',width=1,hatch='////') 

for ov in range(4):
    ax0.scatter(barx_participant[ov]+jitter,p_accept_ov[:,ov,0],edgecolor='black',facecolor='white',alpha=0.3,zorder =  10, s=10,marker='.',linewidth=1)
    ax0.scatter(barx_hm[ov]+jitter,p_accept_ov_ppc_m_hm[:,ov,0],edgecolor='black',facecolor='white',alpha=0.3,zorder =  10, s=10,marker='.',linewidth=1)
    ax0.scatter(barx_sm[ov]+jitter,p_accept_ov_ppc_m_sm[:,ov,0],edgecolor='black',facecolor='white',alpha=0.3,zorder =  10, s=10,marker='.',linewidth=1)
    ax0.scatter(barx_pm[ov]+jitter,p_accept_ov_ppc_m_pm[:,ov,0],edgecolor='black',facecolor='white',alpha=0.3,zorder =  10, s=10,marker='.',linewidth=1)

ax0.set_title('Choice behaviour',fontweight='bold')
ax0.set_xticks(xtick)
ax0.set_xticklabels([1,2,3,4])
ax0.set_ylabel('Proportion "accept"')
ax0.set_xlabel('Offer value')
ax0.set_ylim([-0.03,1.07])
ax0.set_xlim([-1,19])   
ax0.legend(handles=[patch1, patch2, patch3, patch4],loc='upper left', ncol=1,frameon=False,borderpad=0.1,labelspacing = 0.2)

ax0.text(-0.1,1.1,'A',transform=ax0.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')

##############################################################################
# Subplot 2 - Model comparison 
##############################################################################  
y = np.flip(looic.loc[:,['LOOIC']].to_numpy().squeeze())
x = [0,1,2]
yerr = np.flip(looic.loc[:,['SE']].to_numpy().squeeze())

ax1.bar(x,y,yerr = yerr,edgecolor='black',color=[0.7,0.7,0.7])
ax1.set_xticks(x)
ax1.set_xticklabels(['PM','SM','HM'],fontsize = 9,rotation=90)
ax1.set_ylim(3000,5200) 
ax1.text(0.83, 0.4, '*', horizontalalignment='center',verticalalignment='center', transform=ax1.transAxes,fontsize=15,fontweight='bold')
ax1.set_ylabel('LOOIC')
ax1.set_title('Model \n comparison',fontweight='bold')
ax1.text(-0.1,1.1,'B',transform=ax1.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')

##############################################################################
# Subplot 3 - Matched choices between models and participants  
##############################################################################
plt.rcParams['hatch.linewidth'] = 0.5
patch1 = mpatches.Patch(facecolor=[0.7,0.7,0.7], label='PM',edgecolor='black',hatch='////')
patch2 = mpatches.Patch(facecolor=[0.7,0.7,0.7], label='SM',edgecolor='black',hatch='....')
patch3 = mpatches.Patch(facecolor=[0.7,0.7,0.7], label='HM',edgecolor='black',hatch='xxxx')

jitter = np.random.normal(loc= -0.2,scale=0.1,size=ns)
barx_pm = [0,4,8,12]
barx_sm = [0+1,4+1,8+1,12+1]
barx_hm = [0+2,4+2,8+2,12+2]
xtick = [1,5,9,13]
vlines_max_hm= match_hm_ov_m+ match_hm_ov_s
vlines_min_hm = match_hm_ov_m- match_hm_ov_s
vlines_max_sm= match_sm_ov_m + match_sm_ov_s
vlines_min_sm = match_sm_ov_m - match_sm_ov_s
vlines_max_pm= match_pm_ov_m + match_pm_ov_s 
vlines_min_pm = match_pm_ov_m - match_pm_ov_s

ax2.vlines(x=barx_hm,ymin=vlines_min_hm,ymax= vlines_max_hm, color='black', linestyle='-',zorder=20)
ax2.vlines(x=barx_sm,ymin=vlines_min_sm,ymax= vlines_max_sm, color='black', linestyle='-',zorder=20)
ax2.vlines(x=barx_pm,ymin=vlines_min_pm,ymax= vlines_max_pm, color='black', linestyle='-',zorder=20)
ax2.bar(barx_hm,match_hm_ov_m,color=[0.7,0.7,0.7],edgecolor = 'black',width=1,hatch='xxxx') 
ax2.bar(barx_sm,match_sm_ov_m,color=[0.7,0.7,0.7],edgecolor = 'black',width=1,hatch='....') 
ax2.bar(barx_pm,match_pm_ov_m,color=[0.7,0.7,0.7],edgecolor = 'black',width=1,hatch='////') 

ax2.set_xticks(xtick)
ax2.set_xticklabels([1,2,3,4])
ax2.set_ylabel('Proportion matched choices')
ax2.set_xlabel('Offer value')
ax2.set_ylim([0.5,1])
ax2.set_xlim([-1,15])   
ax2.text(-0.1,1.1,'C',transform=ax2.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')

##############################################################################
# Subplot 4 - Parameters of the winning hybrid model 
##############################################################################
param_names_tmp = ['mu_beta_dv','mu_theta_basic_1','mu_theta_basic_2','mu_theta_basic_3','mu_theta_basic_4','theta_full_energy','theta_low_energy_LC','theta_low_energy_HC']
xticklabels = [r'$\beta_{plan}$',r'$\theta_{O1}$',r'$\theta_{O2}$',r'$\theta_{O3}$',r'$\theta_{O4}$',r'$\theta_{maxE}$',r'$\theta_{minE\_LC}$',r'$\theta_{minE\_HC}$']

for i, param in enumerate(param_names_tmp):   
    # Plot PDF
    kernel = stats.gaussian_kde(samples_hybrid[param])
    min_tmp = np.mean(samples_hybrid[param])+np.std(samples_hybrid[param])*4
    max_tmp = np.mean(samples_hybrid[param])-np.std(samples_hybrid[param])*4
    positions = np.linspace(start=min_tmp,stop=max_tmp,num=50)
    pdf_estimate = kernel(positions)
    pdf_estimate = pdf_estimate/ (np.max(pdf_estimate) + 0.3) # scale
    ax3.plot(pdf_estimate + i,positions,color='grey',linewidth=1)
    ax3.fill_betweenx(positions,np.ones(len(positions)) * i,pdf_estimate + i,alpha=0.3,color = 'grey')

    # Plot posterior mean and 95% credibility intervals
    ax3.scatter(i,summary_hybrid[param]['mean'].to_numpy(),color='black',s=10)
    ax3.vlines(i, ymin=summary_hybrid[param]['q2_5'].to_numpy(),ymax=summary_hybrid[param]['q97_5'].to_numpy(),color='black',linewidth=1)

    # Plot posterior mean of participant level estimates 
    jitter = np.random.normal(loc=-0.16,scale=0.05,size=ns)

    if (param == 'mu_theta_basic_1'):
        ax3.scatter(i+jitter,summary_hybrid['theta_basic_1']['mean'],edgecolor='black',facecolor='white',alpha=0.3,s=10,marker='.',linewidth=1,zorder=0)
    elif (param == 'mu_theta_basic_2'):
        ax3.scatter(i+jitter,summary_hybrid['theta_basic_2']['mean'],edgecolor='black',facecolor='white',alpha=0.3,s=10,marker='.',linewidth=1,zorder=0)
    elif (param == 'mu_theta_basic_3'):
        ax3.scatter(i+jitter,summary_hybrid['theta_basic_3']['mean'],edgecolor='black',facecolor='white',alpha=0.3,s=10,marker='.',linewidth=1,zorder=0)
    elif (param == 'mu_theta_basic_4'):
        ax3.scatter(i+jitter,summary_hybrid['theta_basic_4']['mean'],edgecolor='black',facecolor='white',alpha=0.3,s=10,marker='.',linewidth=1,zorder=0)
    elif (param == 'mu_beta_dv'):
        ax3.scatter(i+jitter,summary_hybrid['beta_dv']['mean'],edgecolor='black',facecolor='white',alpha=0.3,s=10,marker='.',linewidth=1,zorder=0)
    elif (param == 'mu_theta_basic'):
        ax3.scatter(i+jitter,summary_hybrid['theta_basic']['mean'],edgecolor='black',facecolor='white',alpha=0.3,s=10,marker='.',linewidth=1,zorder=0)
        
ax3.hlines(0,xmin=-0.5,xmax= 8,color='black',linestyle=':',zorder=0)

ax3.set_title('Estimated parameters:\nHybrid strategy model',fontweight='bold')
ax3.set_xticks(range(len(param_names_tmp)))
ax3.set_xticklabels(xticklabels,rotation=90,x=np.arange(2,len(param_names_tmp)+1),fontsize = 10)
ax3.set_xlim((-0.5,len(param_names_tmp)))
ax3.text(-0.1,1.1,'D',transform=ax3.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')

gsp.GridSpec(nrows=1,ncols=3,figure=fig,wspace=0.1)
plt.tight_layout()
timestr = time.strftime("%Y%m%d%H%M%S")
# fig.savefig('fig2_' + timestr + '.tif', dpi=300, bbox_inches='tight', transparent=False)
# fig.savefig('fig2_' + timestr + '.png', dpi=300, bbox_inches='tight', transparent=False)

