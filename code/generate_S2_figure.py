"""
Florian Ott, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import glob as glob 
import pandas as pd 
import time as time
plt.style.use('./ara.mplstyle')

#%% Load behavioural data
filename = glob.glob('../data/behaviour/data_all_partipants_20210623102746.csv')
dat = pd.read_csv(filename[0],index_col = 0)
unique_vpns = np.unique(dat['vpn'])
ns = len(unique_vpns )
idx = (dat['timeout'] == 0)
dat_subset = dat.loc[idx]

#%% Load individual LOOIC values
filename = glob.glob('../data/model/model_comparison_individual.csv') 
looic_individual = pd.read_csv(filename[0],index_col = 0)

#%% Generate Supplementary Figure 2
# Calculate the predictive accuracy for each participant by summing 
# trialwise pointwise log predictive densities over all trials of a particpant.
looic_individual_hm = looic_individual['Hybrid'].to_numpy()
looic_individual_sm = looic_individual['Simple'].to_numpy()
looic_individual_pm = looic_individual['Planning'].to_numpy()

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
y = [np.mean(loo_vpn_pm),np.mean(loo_vpn_sm), np.mean(loo_vpn_hm)]
x = [0,1,2]
yerr = [np.std(loo_vpn_pm),np.std(loo_vpn_sm), np.std(loo_vpn_hm)]
x2 = np.zeros((3,ns))
x2[0,:] = np.random.normal(loc=0, scale = 0.1,size = ns)
x2[1,:] = np.random.normal(loc=1, scale = 0.1,size = ns)
x2[2,:] = np.random.normal(loc=2, scale = 0.1,size = ns)
y2 = np.zeros((3,ns))
y2[0,:] = loo_vpn_pm
y2[1,:] = loo_vpn_sm
y2[2,:] = loo_vpn_hm

outlier1 = np.where(np.argmin(y2,axis=0) == 0)
outlier2 = np.where(np.argmin(y2,axis=0) == 1)
outlier = np.where(np.argmin(y2,axis=0) != 2)

# Plotting
fig, ax1 = plt.subplots(1,1,figsize=(3,3))

ax1.bar(x,y,edgecolor='black',color=[0.7,0.7,0.7])
ax1.scatter(x2[0,:],y2[0,:],edgecolor='black',facecolor='white',alpha=0.3,zorder =  10, s=50,marker='.',linewidth=1)
ax1.scatter(x2[1,:],y2[1,:],edgecolor='black',facecolor='white',alpha=0.3,zorder =  10, s=50,marker='.',linewidth=1)
ax1.scatter(x2[2,:],y2[2,:],edgecolor='black',facecolor='white',alpha=0.3,zorder =  10, s=50,marker='.',linewidth=1)
for s in range(ns):
    if np.in1d(s, outlier1):
        ax1.plot([x2[0,s],x2[1,s],x2[2,s]], [y2[0,s],y2[1,s],y2[2,s]],color='red',linewidth=0.5)
    elif np.in1d(s, outlier2):
        ax1.plot([x2[0,s],x2[1,s],x2[2,s]], [y2[0,s],y2[1,s],y2[2,s]],color='darkmagenta',linewidth=0.5)
    else:
        ax1.plot([x2[0,s],x2[1,s],x2[2,s]], [y2[0,s],y2[1,s],y2[2,s]],color='black',linewidth=0.5)
        
ax1.set_xticks(x)
ax1.set_xticklabels(['PM','SM','HM'],fontsize = 9,rotation=90)
ax1.set_xlim(-0.5,2.5) 
ax1.set_ylim(30,270) 
ax1.set_ylabel('LOOIC_i')
plt.tight_layout()

# Save Plot
timestr = time.strftime("%Y%m%d%H%M%S")
# fig.savefig('S1_fig_' + timestr + '.tif', dpi=300, bbox_inches='tight', transparent=False)
# fig.savefig('S1_fig_' + timestr + '.png', dpi=300, bbox_inches='tight', transparent=False)
