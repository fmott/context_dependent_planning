"""
Florian Ott, 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import glob as glob 
import pandas as pd
import scipy.stats as stats
 
plt.style.use('./ara.mplstyle')

#%% Load behavioural data 
filename = glob.glob('../data/behaviour/data_all_participants_20220215120148.csv')
dat = pd.read_csv(filename[0],index_col = 0)

#%% Rebuttal letter figures - Geometric distributions of offer occurences 
# Offers across index
idx = (dat['vpn']==101) & (dat['reward']==1) 
indices1 = dat.loc[idx,['index']].to_numpy()
idx = (dat['vpn']==101) & (dat['reward']==2) 
indices2 = dat.loc[idx,['index']].to_numpy()
idx = (dat['vpn']==101) & (dat['reward']==3) 
indices3 = dat.loc[idx,['index']].to_numpy()
idx = (dat['vpn']==101) & (dat['reward']==4) 
indices4 = dat.loc[idx,['index']].to_numpy()

# Index difference distributions
diff1 = indices1[1:-1] - indices1[0:-2]
diff2 = indices2[1:-1] - indices2[0:-2]
diff3 = indices3[1:-1] - indices3[0:-2]
diff4 = indices4[1:-1] - indices4[0:-2]
diffs = [diff1,diff2,diff3,diff4]

# Counts
counts1 = np.array([np.sum(diff1==i) for i in range(1,16)])/np.sum(np.array([np.sum(diff1==i) for i in range(1,16)]))
counts2 = np.array([np.sum(diff2==i) for i in range(1,16)])/np.sum(np.array([np.sum(diff2==i) for i in range(1,16)]))
counts3 = np.array([np.sum(diff3==i) for i in range(1,16)])/np.sum(np.array([np.sum(diff3==i) for i in range(1,16)]))
counts4 = np.array([np.sum(diff4==i) for i in range(1,16)])/np.sum(np.array([np.sum(diff4==i) for i in range(1,16)]))
counts = [counts1,counts2, counts3, counts4 ]

# Geometric dist
x = np.arange(1,16)
x2 = np.arange(1,16)-0.2

# Plotting
fig, ax = plt.subplots(2,2,figsize=(5,3.5))
for i,axes in enumerate(ax.flat):
    axes.plot(x2,counts[i], 'ro', ms=2, alpha=0.5,label='empirical')
    h1 = axes.vlines(x2, 0, counts[i], colors='red', lw=1, alpha=0.5,label='empirical')

    axes.plot(x, stats.geom.pmf(x, 0.25), 'bo', ms=2, alpha=0.5)
    h2 = axes.vlines(x, 0, stats.geom.pmf(x, 0.25), colors='b', lw=1, alpha=0.5,label='geometric \np=0.25')
    axes.set_title('o = '+str(i+1))
    axes.set_xlim(0,15)
    axes.set_xticks([5,10,15])
    
    if i >1:
        axes.set_xlabel('# Trials between \n consecutive occurences',fontsize=8)
    if (i == 0) | (i == 2):
        axes.set_ylabel('Normalised Count',fontsize=8)
    
    if i ==0: 
        # axes.legend(handles=[h1, h2],frameon=False,loc ='center', bbox_to_anchor=(0.7, 0.6),fontsize=8)
        axes.legend(handles=[h1, h2],frameon=False,fontsize=8)


plt.tight_layout()
# fig.savefig('S6_fig.tif', dpi=300, bbox_inches='tight', transparent=False)
