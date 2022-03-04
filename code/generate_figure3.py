"""
Florian Ott, 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import glob as glob 
import pandas as pd 
import time as time
import scipy.stats as stats
import pickle as pickle 

plt.style.use('./ara.mplstyle')

#%% Load data
## Behaviour 
filename = glob.glob('../data/behaviour/data_all_participants_20220215120148.csv')
dat = pd.read_csv(filename[0],index_col = 0)

idx = dat['timeout'] == 0
dat_subset = dat.loc[idx]

unique_vpns = np.unique(dat['vpn'])
ns = len(unique_vpns)

## Summary of hybrid model 
with open('../data/model/summary_HM_20220216143030.pkl', 'rb') as f:
    summary_hybrid = pickle.load(f)['summary']

## Indindividual LOOIC values
filename = glob.glob('../data/model/model_comparison_individual.csv') 
looic_individual = pd.read_csv(filename[0])
names = list(looic_individual.columns)
#%% Plot
fig = plt.figure(constrained_layout=False,figsize=(5.1,3.9))

gs = fig.add_gridspec(48, 48)
ax0 = fig.add_subplot(gs[0:18,0:20])
ax1 = fig.add_subplot(gs[0:18,28:48])
ax2 = fig.add_subplot(gs[27:48,0:16])
ax3 = fig.add_subplot(gs[27:48,22:32])
ax4 = fig.add_subplot(gs[27:48,38:48])

##############################################################################
# Subplot 1 and 2 - Post Hoc correlations - Points vs beta_plan
##############################################################################
# Get posterior mean of partipant level planning weights
y1 = summary_hybrid['beta_dv_23']['mean'].to_numpy().squeeze()
y2 = summary_hybrid['beta_dv_14']['mean'].to_numpy().squeeze()

# Get total accumulated points for each participant
idx = dat['index'] == 239
points = dat.loc[idx,['points_after']].to_numpy().squeeze() 
rt = dat.groupby('vpn')['reaction_time'].mean().to_numpy()

# Calculate correlation and regression line 
pars1 = np.polyfit(points,y1, 1)
xx1 = np.arange(np.min(points),np.max(points),0.1)
yy1 = pars1[0]*xx1 + pars1[1]
r1,p1 = stats.pearsonr(points,y1)

pars2 = np.polyfit(points,y2, 1)
xx2 = np.arange(np.min(points),np.max(points),0.1)
yy2 = pars2[0]*xx2 + pars2[1]
r2,p2 = stats.pearsonr(points,y2)

r3,p3 = stats.pearsonr(rt,y1)
r4,p4 = stats.pearsonr(rt,y2)

# Plotting
ax0.scatter(points,y1,color='black',edgecolor='black',facecolor='white',alpha=0.6,zorder=10,s=20,marker='.',linewidth=1)
ax0.plot(xx1,yy1,color='black',linewidth=1)
ax0.set_ylabel(r'$\beta_{plan23}$',fontsize = 12)
ax0.set_xlabel('Points')
ax0.text(-0.1,1.1,'A',transform=ax0.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')

ax1.scatter(points,y2,color='black',edgecolor='black',facecolor='white',alpha=0.6,zorder=10,s=20,marker='.',linewidth=1)
ax1.plot(xx2,yy2,color='black',linewidth=1)
ax1.set_ylabel(r'$\beta_{plan14}$',fontsize = 12)
ax1.set_xlabel('Points')
ax1.text(-0.1,1.1,'B',transform=ax1.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')

##############################################################################
# Subplot 3 - Particpant level model performances
##############################################################################
# Calculate the predictive accuracy for each participant by summing 
# trialwise pointwise log predictive densities over all trials of a particpant.
looic_individual_hm = looic_individual['HM (I)P+S (E)P+S 2beta'].to_numpy()
looic_individual_sm = looic_individual['SM (I)S (E)S'].to_numpy()
looic_individual_pm = looic_individual['PM (I)P (E)P'].to_numpy()

loo_vpn_hm = np.zeros(ns)
loo_vpn_sm = np.zeros(ns)
loo_vpn_pm = np.zeros(ns)

for i, vpn in enumerate(unique_vpns):
    idx = dat_subset['vpn'] == vpn 
    loo_vpn_hm[i] = np.sum(looic_individual_hm[idx])
    loo_vpn_sm[i] = np.sum(looic_individual_sm[idx])
    loo_vpn_pm[i] = np.sum(looic_individual_pm[idx])
                                               
# Preparing data for Plotting  
np.random.seed(1)   
y = [np.mean(loo_vpn_sm),np.mean(loo_vpn_pm), np.mean(loo_vpn_hm)]
x = [0,1,2]
yerr = [np.std(loo_vpn_sm),np.std(loo_vpn_pm), np.std(loo_vpn_hm)]
x2 = np.zeros((3,ns))
x2[0,:] = np.random.normal(loc=0, scale = 0.1,size = ns)
x2[1,:] = np.random.normal(loc=1, scale = 0.1,size = ns)
x2[2,:] = np.random.normal(loc=2, scale = 0.1,size = ns)
y2 = np.zeros((3,ns))
y2[0,:] = loo_vpn_sm
y2[1,:] = loo_vpn_pm
y2[2,:] = loo_vpn_hm

outlier1 = np.where(np.argmin(y2,axis=0) == 0)
outlier2 = np.where(np.argmin(y2,axis=0) == 1)
outlier = np.where(np.argmin(y2,axis=0) != 2)

# Plotting
ax2.bar(x,y,edgecolor='black',color=[0.7,0.7,0.7])
ax2.scatter(x2[0,:],y2[0,:],edgecolor='black',facecolor='white',alpha=0.4,zorder =  10, s=20,marker='.',linewidth=1)
ax2.scatter(x2[1,:],y2[1,:],edgecolor='black',facecolor='white',alpha=0.4,zorder =  10, s=20,marker='.',linewidth=1)
ax2.scatter(x2[2,:],y2[2,:],edgecolor='black',facecolor='white',alpha=0.4,zorder =  10, s=20,marker='.',linewidth=1)
for s in range(ns):
    if np.in1d(s, outlier1):
        ax2.plot([x2[0,s],x2[1,s],x2[2,s]], [y2[0,s],y2[1,s],y2[2,s]],color='red',linewidth=0.5)
    elif np.in1d(s, outlier2):
        ax2.plot([x2[0,s],x2[1,s],x2[2,s]], [y2[0,s],y2[1,s],y2[2,s]],color='darkmagenta',linewidth=0.5)
    else:
        ax2.plot([x2[0,s],x2[1,s],x2[2,s]], [y2[0,s],y2[1,s],y2[2,s]],color='black',linewidth=0.5)
        
ax2.set_xticks(x)
ax2.set_xticklabels(['SM','PM','HM'],fontsize = 9,rotation=90)
ax2.set_xlim(-0.5,2.5) 
ax2.set_ylim(30,230) 
ax2.set_ylabel('LOOIC_i')
ax2.text(-0.1,1.1,'C',transform=ax2.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')


##############################################################################
# Subplot 3 and 4 -  Median split of RTs and Points based on the 
# the particpant-level difference of PM minus SM LOOIC_i
##############################################################################

loo_vpn_sm = np.zeros(ns)
loo_vpn_pm = np.zeros(ns)
loo_vpn_hm = np.zeros(ns)

for i, vpn in enumerate(unique_vpns):
    idx = dat_subset['vpn'] == vpn 
    loo_vpn_sm[i] = np.sum(looic_individual_sm[idx])
    loo_vpn_pm[i] = np.sum(looic_individual_pm[idx])
    loo_vpn_hm[i] = np.sum(looic_individual_hm[idx])
                                      
loo_vpn_all = np.array([loo_vpn_sm, loo_vpn_pm,loo_vpn_hm]).T

loo_pm_diff_sm = loo_vpn_all[:,1] - loo_vpn_all[:,0] # small number means more evidence for pm
vpn_pm_support = unique_vpns[np.where(loo_pm_diff_sm  <= np.median(loo_pm_diff_sm))]
vpn_sm_support = unique_vpns[np.where(loo_pm_diff_sm  > np.median(loo_pm_diff_sm))]

## Individual performance dependening on best fit
# Accumulated points
idx = np.isin(dat['vpn'],vpn_sm_support) & (dat['index']==239)
points_sm_support = dat.loc[idx,['points_after']].to_numpy().squeeze()
points_sm_support_m= np.mean(points_sm_support)
points_sm_support_s = np.std(points_sm_support)

idx = np.isin(dat['vpn'],vpn_pm_support) & (dat['index']==239)
points_pm_support = dat.loc[idx,['points_after']].to_numpy().squeeze()
points_pm_support_m = np.mean(points_pm_support)
points_pm_support_s = np.std(points_pm_support)

## Plotting
barx = [0,1]
bary = [points_sm_support_m,points_pm_support_m]
barerr = [points_sm_support_s,points_pm_support_s]
xticks = [0,1]
xticklabels = ['SM', 'PM']
scatterx = np.repeat(np.array([0,1]), [len(points_sm_support),len(points_pm_support)])+np.random.normal(0.1,0.05, size=40)
scattery = np.concatenate([points_sm_support,points_pm_support])
ax3.bar(barx,bary,yerr = barerr,edgecolor='black',color=[0.7,0.7,0.7])
ax3.scatter(scatterx, scattery,edgecolor='black',facecolor='white',alpha=0.6,zorder =  10, s=20,marker='.',linewidth=1)
ax3.set_ylim(320,350)
ax3.set_ylabel('Points',labelpad=0)
ax3.set_xticks(xticks)
ax3.set_xticklabels(xticklabels,fontsize = 9,rotation=90)
ax3.text(-0.1,1.1,'D',transform=ax3.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')



## Reaction time
rt_sm_support_m = np.zeros(len(vpn_sm_support))
for s, vpn in enumerate(vpn_sm_support):
    idx = (dat['vpn'] == vpn)
    rt_sm_support_m[s] = np.mean(dat.loc[idx,['reaction_time']])
    
rt_sm_support_mm= np.nanmean(rt_sm_support_m)
rt_sm_support_ms = np.nanstd(rt_sm_support_m)

rt_pm_support_m = np.zeros(len(vpn_pm_support))
for s, vpn in enumerate(vpn_pm_support):
    idx = (dat['vpn'] == vpn)
    rt_pm_support_m[s] = np.mean(dat.loc[idx,['reaction_time']])
    
rt_pm_support_mm= np.nanmean(rt_pm_support_m)
rt_pm_support_ms = np.nanstd(rt_pm_support_m)

## Plotting
barx = [0,1]
bary = [rt_sm_support_mm,rt_pm_support_mm]
barerr = [rt_sm_support_ms,rt_pm_support_ms,]
scatterx = np.repeat(np.array([0,1]), [len(rt_sm_support_m),len(rt_pm_support_m)])+np.random.normal(0.1,0.05, size=40)
scattery = np.concatenate([rt_sm_support_m, rt_pm_support_m])
ax4.bar(barx,bary,yerr = barerr,edgecolor='black',color=[0.7,0.7,0.7])
ax4.scatter(scatterx, scattery,edgecolor='black',facecolor='white',alpha=0.6,zorder =  10, s=20,marker='.',linewidth=1)
ax4.set_ylim(0.3,1.8)
ax4.set_ylabel('RT [s]',labelpad=0)
ax4.set_xticks(xticks)
ax4.set_xticklabels(xticklabels,fontsize = 9,rotation=90)
ax4.text(-0.1,1.1,'E',transform=ax4.transAxes,horizontalalignment='center',verticalalignment='center',fontsize=10,fontweight='bold')
ax4.hlines(1.65,0,1,linewidth=0.5)
ax4.text(0.5, 0.93, '*', horizontalalignment='center',verticalalignment='center', transform=ax4.transAxes,fontsize=10,fontweight='bold')

## Significance testing 
print(stats.mannwhitneyu(points_pm_support,points_sm_support))
print(stats.mannwhitneyu(rt_pm_support_m,rt_sm_support_m))
timestr = time.strftime("%Y%m%d%H%M%S")
# fig.savefig('S3_fig_' + timestr + '.tif', dpi=1000, bbox_inches='tight', transparent=False)