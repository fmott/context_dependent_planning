"""
Florian Ott, 2022
"""

import numpy as np
import matplotlib.pyplot as plt
import glob as glob 
import pandas as pd 
import time as time
plt.style.use('./ara.mplstyle')

#%% Load behavioural data 
filename = glob.glob('../data/behaviour/data_all_participants_20220215120148.csv')
dat = pd.read_csv(filename[0],index_col = 0)

#%% LOOIC for model comparison plot
filename = glob.glob('../data/model/model_comparison_alternative_SM.csv') 
looic_df = pd.read_csv(filename[0],index_col = 0)
looic = looic_df['LOOIC'].to_numpy()
looic_se = looic_df['SE'].to_numpy()
names = list(looic_df.index)

#%% Generate model comparison plot
sort_idx = np.argsort(looic)
barx = np.array(range(len(looic),0,-1   )  )
bary = looic[sort_idx]
barerr = looic_se[sort_idx]
bar_name = [names[i] for i in sort_idx]

fig,ax = plt.subplots(1,1,figsize=(4,1))

ax.barh(barx, bary, xerr=barerr,color='grey')
ax.set_yticks(barx)
ax.set_yticklabels(bar_name,rotation=0)
ax.set_xlabel('LOOIC')
ax.set_xlim(4200,6000) 

timestr = time.strftime("%Y%m%d%H%M%S")
# fig.savefig('S2_fig_' + timestr + '.tif', dpi=300, bbox_inches='tight', transparent=False)