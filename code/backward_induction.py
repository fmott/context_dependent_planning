"""
Florian Ott, 2020
"""

import numpy as np

def ara_backward_induction(energy_cost_current = 1,energy_cost_future = 1,energy_bonus = 0):
    """
    This function implements the backward induction algorithm.
    The inputs energy_cost_future and energy_cost_current specify the energy 
    costs of the current and future segments. Energy bonus specifies how much 
    value is assigned per remaining energy unit after the final trial.
    """
    
    n_energy            = 7 # number of energy states
    n_offer             = 4 # number of offers
    n_action            = 2 # number of actions
    n_trial             = 8 # number of trials 
    max_energy          = 6 # maximum energy state
    op                  = np.array([0.25,0.25,0.25,0.25]) # offer probability
    
    V = np.zeros((n_energy,n_offer,n_trial+1)) # state value function 
    Q = np.zeros((n_energy,n_offer,n_action,n_trial)) # state action value function
    
    final_reward = np.tile(np.arange(n_energy),(n_offer,1)).T * energy_bonus # terminal reward 
    V[:,:,0] = final_reward
    
    # loop through the statespace and timesteps
    # Note1: If speed is needed, vectorize and use tranision matrix 
    # Note2: Since we are implicitly looping backwards, the future 
    # segment comes first
    for t in range(n_trial):
        for e in range(n_energy):
            for a in range(n_action):               
                for o in range(n_offer):
    
                    if t < n_trial/2: # Future segment
                        if a == 0: # accept
                            if e >= energy_cost_future: # enough energy
                                Q[e,o,a,t] = o+1 + np.sum(V[e-energy_cost_future,:,t]*op)
                            elif e < energy_cost_future: # not enough energy
                                Q[e,o,a,t] = np.sum(V[0,:,t]*op) # energy goes to 0
                        elif a == 1: # reject
                            if e < max_energy: # not max energy
                                Q[e,o,a,t] = np.sum(V[e+1,:,t]*op)
                            if e == max_energy: # max energy
                                Q[e,o,a,t] = np.sum(V[e,:,t]*op)
                            
                    elif t >= n_trial/2:  # Current segment 
                        if a == 0: # accept
                            if e >= energy_cost_current: # enough energy
                                Q[e,o,a,t] = o+1 + np.sum(V[e-energy_cost_current,:,t]*op)
                            elif e < energy_cost_current: # not enough energy
                                Q[e,o,a,t] = np.sum(V[0,:,t]*op) # energy goes to 0
                        elif a == 1: # reject
                            if e < max_energy: # not max energy
                                Q[e,o,a,t] = np.sum(V[e+1,:,t]*op)
                            if e == max_energy: # max energy
                                Q[e,o,a,t] = np.sum(V[e,:,t]*op)
                
                    # Value of the maximizing action given energy state e,
                    # offer o and trial t 
                    Qmax = np.maximum(Q[e,o,0,t],Q[e,o,1,t])
                               
                    # Update state value function  
                    V[e,o,t+1] = Qmax
    
    # Flip V and Q such that the first trial corresponds to te first index  
    V = V[:,:,::-1]
    Q = Q[:,:,:,::-1]   

    return V, Q         
            

#%%  Run backward induction
# Calculate decision varaible DV and conflict C
costs = np.array([[1,1], [2,1],[1,2],[2,2]]) # The four possible transitions
energy_bonus=2.5/1.5
V = np.zeros((7,4,9,4)) # State-value function 
Q = np.zeros((7,4,2,8,4)) # Stat-action function 

for i in range(4):
    V[:,:,:,i], Q[:,:,:,:,i] = ara_backward_induction(energy_cost_current = costs[i][0],energy_cost_future = costs[i][1],energy_bonus=energy_bonus)
DV = Q[:,:,0,:,:] - Q[:,:,1,:,:] 
C = np.absolute(DV) * -1

