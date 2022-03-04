"""
Florian Ott, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import glob as glob 
import pandas as pd 
import matplotlib.patches as mpatches
plt.style.use('./ara.mplstyle')

#%% Load behavioural data 
filename = glob.glob('../data/behaviour/data_all_participants_20220215120148.csv')
dat = pd.read_csv(filename[0],index_col = 0)

#%% Generate Supplementary Figure 5
# Get conflict distributions for different offer values
conflict_ov = [None] * 4 
for ov in range(4):
    idx = (dat['reward'] == (ov+1))
    conflict_ov[ov] = dat.loc[idx,['conflict_planning']].to_numpy().squeeze()

# Get conflict distributions for the intermediate and extreme context
conflict_14 = np.concatenate((conflict_ov[0],conflict_ov[3])) 
conflict_23 = np.concatenate((conflict_ov[1],conflict_ov[2])) 

# Plotting 
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=[4,3])
patch1 = mpatches.Patch(facecolor='darkorange',label='Offer 1 and 4',edgecolor=None)
patch2 = mpatches.Patch(facecolor='navy',label='Offer 2 and 3',edgecolor=None)
ax.set_ylim((0,750))

ax.hist(conflict_14,facecolor='darkorange',alpha=0.7,bins=12)
ax.hist(conflict_23,facecolor='navy',alpha=0.7,bins=12)
ax.vlines(np.mean(conflict_14),ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],color='darkorange',linewidth = 1,linestyle ='--')
ax.vlines(np.mean(conflict_23),ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],color='navy',linewidth = 1,linestyle ='--')

ax.set_xlabel('Conflict')
ax.set_ylabel('# Trials')        
ax.legend(handles=[patch1, patch2],loc='upper center',ncol=2,bbox_to_anchor=(0.5, 1.15),frameon=False)

plt.tight_layout()
# fig.savefig('S5_fig.tif', dpi=300, bbox_inches='tight', transparent=False)
