"""
Florian Ott, 2020
"""

import numpy as np
import pandas as pd
import glob as glob
import time as time 
import pickle 
from backward_induction import ara_backward_induction

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#%% Load model data
# Posterior mean and credibility intervals of estimated parameters
with open("../data/model/summary_planning_20210629104942.pkl", "rb") as f:
    summary_planning = pickle.load(f)['summary']

with open("../data/model/summary_simple_20210621151244.pkl", "rb") as f:
    summary_simple = pickle.load(f)['summary']

with open("../data/model/summary_hybrid_20210621151244.pkl", "rb") as f:
    summary_hybrid = pickle.load(f)['summary']


#%% Load participant data
filename = glob.glob('../data/behaviour/data_all_partipants_20210623102746.csv')
dat = pd.read_csv(filename[0],index_col = 0)
nt = len(np.unique(dat['index']))


#%% Select agent
# agent_name = 'planning'
# agent_name = 'simple'
agent_name = 'hybrid'
# agent_name = 'random'

if agent_name =='planning':
    # Calculate DV based on backward induction. 
    # Columns of DV are energy, offer value, trial, transition.
    costs = np.array([[1,1], [2,1],[1,2],[2,2]]) # The four possible transitions
    V = np.zeros((7,4,9,4)) # State-value function 
    Q = np.zeros((7,4,2,8,4)) # Stat-action function 

    for i in range(4):
        V[:,:,:,i], Q[:,:,:,:,i] = ara_backward_induction(energy_cost_current = costs[i][0],energy_cost_future = costs[i][1],energy_bonus=(2.5/1.5))
    
    DV = Q[:,:,0,:,:] - Q[:,:,1,:,:] 
    
    beta = summary_planning['mu_beta_dv']['mean'].to_numpy().squeeze()
    bias = summary_planning['mu_theta_basic']['mean'].to_numpy()

elif agent_name =='simple':
    # DV corresponds to centered offer value
    DV = np.zeros((7,4,8,4))
    DV[:,0,:,:] = -1.5
    DV[:,1,:,:] = -0.5
    DV[:,2,:,:] = 0.5
    DV[:,3,:,:] = 1.5
    
    beta = summary_simple['mu_beta_dv']['mean'].to_numpy().squeeze()
    bias = summary_simple['mu_theta_basic']['mean'].to_numpy()
    
  
if agent_name =='hybrid':
    # Calculate DV based on backward induction. 
    # Columns of DV are energy, offer value, trial, transition.
    costs = np.array([[1,1], [2,1],[1,2],[2,2]]) # The four possible transitions
    V = np.zeros((7,4,9,4)) # State-value function 
    Q = np.zeros((7,4,2,8,4)) # Stat-action function 

    for i in range(4):
        V[:,:,:,i], Q[:,:,:,:,i] = ara_backward_induction(energy_cost_current = costs[i][0],energy_cost_future = costs[i][1],energy_bonus=(2.5/1.5))
    
    DV = Q[:,:,0,:,:] - Q[:,:,1,:,:] 
    
    beta = summary_hybrid['mu_beta_dv']['mean'].to_numpy().squeeze()
    bias1 = summary_hybrid['mu_theta_basic_1']['mean'].to_numpy()  
    bias2 = summary_hybrid['mu_theta_basic_2']['mean'].to_numpy()  
    bias3 = summary_hybrid['mu_theta_basic_3']['mean'].to_numpy()  
    bias4 = summary_hybrid['mu_theta_basic_4']['mean'].to_numpy()  
    bias = [bias1,bias2,bias3,bias4]
    
elif agent_name =='random':
    DV = np.zeros((7,4,8,4))
    beta = 0 
    bias = 0


#%% Get stimulus features and preallocate variables
idx = dat['vpn'] == 101
response = np.zeros(nt,dtype=int)
points = np.zeros(nt,dtype=int)
points_after = np.zeros(nt,dtype=int)
energy = np.zeros(nt,dtype=int)
energy_after = np.zeros(nt,dtype=int)
reward = dat.loc[idx,['reward']].to_numpy().squeeze()
energy_cost = dat.loc[idx,['energy_cost']].to_numpy().squeeze()
trial = dat.loc[idx,['trial']].to_numpy().squeeze()
segment= dat.loc[idx,['segment']].to_numpy().squeeze()
segment_after = dat.loc[idx,['segment_after']].to_numpy().squeeze()
transition = dat.loc[idx,['transition']].to_numpy().squeeze()
timeout = np.zeros(nt,dtype=int)
dv = np.zeros(nt)
index = dat.loc[idx,['index']].to_numpy().squeeze()
ns = 200 # number of simulated agents
vpn = np.zeros(nt,dtype=int)
points[0] = 0 # initial points
energy[0] = 3 # initial energy


#%% Simulate agent
agdat = pd.DataFrame()
for s in range(ns): # vectorize if speed is needed
    for t in range(nt):
        if t > 0:
            points[t] = points_after[t-1] # update point score
            energy[t] = energy_after[t-1] # update energy score
        
        # action selection
        dv[t] = DV[energy[t],reward[t]-1,trial[t],transition[t]] # get current decision variable  
             
        if agent_name == 'hybrid':
            choice_probability = sigmoid(beta*dv[t] + bias[reward[t]-1]) # get choice probability
        else: 
            choice_probability = sigmoid(beta*dv[t] + bias) 
            
        if energy[t] == 6:
            response[t] = 0
        elif (energy[t] < 1) & (segment[t] == 0):
            response[t] = 1
        elif (energy[t] < 2) & (segment[t] == 1):
            response[t] = 1
        else:
            response[t] = (np.random.binomial(n=1,p=choice_probability) == 0) # sample choice   

        
        # update state
        if response[t] == 0: # accept            
            if segment[t] == 0: # LC context
                energy_after[t] =  np.clip((energy[t] - 1),a_min=0,a_max=6) # decrease energy by 1, but not below zero
                
                if (energy[t] >= 1):
                    points_after[t] = points[t] + reward[t] # only increase points, if at least 1 energy 
                else:
                    points_after[t] = points[t]
                    
            if segment[t] == 1: # HC context
                energy_after[t] =  np.clip((energy[t] - 2),a_min=0,a_max=6) # decrease energy by 2, but not below zero
                
                if (energy[t] >= 2):
                    points_after[t] = points[t] + reward[t]  # only increase points, if at least 2 energy 
                else:
                    points_after[t] = points[t]         
                
        elif response[t] == 1:# wait
            energy_after[t] = np.clip((energy[t] + 1),a_min=0,a_max=6) # add 1 energy
            points_after[t] = points[t]
            
        vpn[t] = s
            
            
# Fill data frame           
    agdat_tmp = pd.DataFrame({'response':response,
                  'points':points,
                  'points_after':points_after,
                  'energy':energy,
                  'energy_after':energy_after,
                  'reward':reward,
                  'energy_cost':energy_cost,
                  'trial':trial,                             
                  'segment':segment,
                  'segment_after':segment_after,
                  'transition':transition,
                  'timeout':timeout,
                  'vpn':vpn,
                  'dv':dv,
                  'index':index                    
                      }) 
    
    agdat = agdat.append(agdat_tmp)
 
timestr = time.strftime("%Y%m%d%H%M%S") 
agdat.to_csv('sim_'+agent_name+'_posterior_mean_beta_bias_'+timestr+'.csv')    