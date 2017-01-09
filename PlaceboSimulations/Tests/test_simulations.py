# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 17:02:27 2017

@author: jb2428
"""

from __future__ import division



from sys import path
path.append(r'C:\Users\jb2428\Desktop\python\cer\PlaceboSimulations')
from treatment_effect_under_placebo_simulations import scenario_B
import numpy as np
import time


##############################################################################
# Test Scenario B time
##############################################################################

# Run all methods
nlocs = 10
locs = np.linspace(-2,2,nlocs)
multiplier = (200)*(1000)/(10*10.)    # nlocs * n_iter / n_iter_in_evaluation

start = time.time()
cis = np.zeros((nlocs,2))
method = 'regression'
for i, loc in enumerate(locs):        
    cis[i,:], num_valid = scenario_B(N = 100,
                                     n_iter = 10,
                                     loc = loc,
                                     distribution_type = 'uniform',
                                     method = method)        
print('\n  ', multiplier * (time.time()-start))
# 3 minutes

    
start = time.time() 
cis = np.zeros((nlocs,2)) 
method = 'matchdist'
for i, loc in enumerate(locs):        
    cis[i,:], num_valid = scenario_B(N = 100,
                                     n_iter = 10,
                                     loc = loc,
                                     distribution_type = 'uniform',
                                     method = method)     
print('\n  ',multiplier * (time.time()-start))
# 47 hours


##############################################################################
# Scenario B testing function
##############################################################################
"""
from treatment_effect_under_placebo_simulations import generate_binomial_outcome
from treatment_effect_under_placebo_simulations import match_dist
from treatment_effect_under_placebo_simulations import treatment_effect_estimator

def scenario_B_testing(N = 100,
                       n_iter = 1000,
                       loc = 1.,
                       distribution_type = 'uniform',
                       method = None,
                       slice_flag=True,
                       te_method='gcomp',
                       **kwargs):

    Scenario B. Continuous risk factor; control/treated offset by distribution location 
      
        
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2')  
    
    if 'caliper' in kwargs:
        caliper = kwargs['caliper']
    else:
        caliper = 0.2 
    
    estimated_treatment_effect = np.zeros((n_iter,)) 
    treat = np.hstack([np.zeros((N,)),np.ones((N,))])
    nvalid = 0  # number of "valid" runs
    for itera in range(n_iter):   
        if itera % 100 == 0:
            print('.',end='')
        elif itera % 1000 == 0:
            print('.',end='\n')
        # generate the risk factor   
        if distribution_type == 'gaussian':        
            # x = np.vstack([np.random.normal(loc=0.,scale=1.,size=(N,1)), np.random.normal(loc=loc,scale=1.,size=(N,1))])    
            x = np.hstack([np.random.normal(loc=0.,scale=1.,size=(N,)), np.random.normal(loc=loc,scale=1.,size=(N,))])    
        elif distribution_type == 'uniform':
            # x = np.vstack([np.random.uniform(low=0.,high=1.,size=(N,1)), np.random.uniform(low=loc,high=loc+1.,size=(N,1))]) 
            x = np.hstack([np.random.uniform(low=0.,high=1.,size=(N,)), np.random.uniform(low=loc,high=loc+1.,size=(N,))])  
        else:
            raise ValueError('distribution_type [{}] not defined.'.format(distribution_type)) 
        # X = np.hstack([x, treat])
        X = np.vstack([x, treat]).T
        y = generate_binomial_outcome(X[:,0]) # generate the outcome
        if method == 'matchdist':            
            idx = match_dist(x, treat, caliper, slice_flag)              
            X = X[idx,:]
            y = y[idx]
        try:
            clf.fit(X,y)
            estimated_treatment_effect[itera] = treatment_effect_estimator(X,clf,method=te_method)
            nvalid = nvalid+1
        except ValueError:            
            estimated_treatment_effect[itera] = 0.        
    cis = [np.percentile(estimated_treatment_effect,2.5), np.percentile(estimated_treatment_effect,97.5)]     
    return cis, nvalid
"""    
    
##############################################################################
# Test Scenario B alternatives
##############################################################################



cis = np.zeros((nlocs,2)) 
for te_method in ['gcomp', 'gcomp_slow']:    
    start = time.time() 
    for i, loc in enumerate(locs):                    
        cis[i,:], num_valid = scenario_B(N = 100,
                                         n_iter = 100,
                                         loc = loc,
                                         distribution_type = 'uniform',
                                         method = 'matchdist',
                                         te_estimator=te_method)
    print('\n  ',str(te_method), multiplier * (time.time()-start))

"""
.
   test False 195288.348197937
.
   test True 381549.2630004883
.
   gcomp False 578725.733757019
.
   gcomp True 763471.8465805054      
   
"""   