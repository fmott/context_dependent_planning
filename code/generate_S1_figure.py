"""
Florian Ott, 2022
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import glob as glob 
import pandas as pd 
import time as time
plt.style.use('./ara.mplstyle')

#%% Load behavioural data 
filename = glob.glob('../data/behaviour/data_all_participants_20220215120148.csv')
dat = pd.read_csv(filename[0],index_col = 0)

#%% LOOIC for model comparison plot
filename = glob.glob('../data/model/model_comparison_feature_contingent_planning.csv') 
looic_df = pd.read_csv(filename[0],index_col = 0)
looic = looic_df['LOOIC'].to_numpy()
looic_se = looic_df['SE'].to_numpy()
names_tmp = list(looic_df.index)
prefix = ['','','','','','d)  ','c)  ','b)  ','a)  ']
names = [prefix[i] + names_tmp[i] for i in range(len(names_tmp))]

filename = glob.glob('../data/model/model_comparison_feature_contingent_planning_dloo.csv') 
looic_df = pd.read_csv(filename[0],index_col = 0)
d_looic = looic_df['d_LOOIC'].to_numpy()
d_looic_q2_5 = looic_df['d_q2_5'].to_numpy()
d_looic_q97_5 = looic_df['d_q97_5'].to_numpy()

#%% Generate model comparison plot
sort_idx = np.argsort(looic)
barx = np.array(range(len(looic),0,-1   )  )
bary = looic[sort_idx]
barerr = looic_se[sort_idx]
bar_name = [names[i] for i in sort_idx]

fig,ax = plt.subplots(1,1,figsize=(4,2))

ax.barh(barx, bary, xerr=barerr,color='grey')
ax.set_yticks(barx)
ax.set_yticklabels(bar_name,rotation=0)
ax.set_xlabel('LOOIC')
ax.set_xlim(3000,4800) 

axins = inset_axes(ax, width=0.8, height=0.55)
axins.vlines(0,ymin=-0.5,ymax=3.5, color='red',linestyle='--',linewidth=1)
axins.scatter(d_looic, [0,1,2,3],edgecolor='black',facecolor='black',alpha=1,zorder =  10, s=20,marker='.',linewidth=1)
axins.hlines([0,1,2,3],xmax=d_looic_q97_5,xmin=d_looic_q2_5,linewidth = 1)
axins.set_xlabel('d_LOOIC',labelpad=0)
axins.set_yticks([0,1,2,3])
axins.set_xticks([0,25,50,75])
axins.set_yticklabels(['c vs. d', 'b vs. c', 'a vs. b', 'a vs. c'],fontsize=8)

timestr = time.strftime("%Y%m%d%H%M%S")
# fig.savefig('S1_fig_' + timestr + '.tif', dpi=300, bbox_inches='tight', transparent=False)











