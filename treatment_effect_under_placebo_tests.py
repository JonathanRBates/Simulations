# -*- coding: utf-8 -*-
"""
Treatment Effect Tests (Pseudo/Placebo Treatments)

One binary treatment with no effect on outcome. One risk factor.

Scenario A. Binary risk factor.

Scenario B. Continuous risk factor, offset by distribution location between control/treated

Scenario C. Continuous risk factor, treatment correlated to risk factor.
"""

import numpy as np

##############################################################################
# Scenario A. Binary risk factor.
##############################################################################

def scenario_A_match_exact(x, treat):
    """
    Produce Exact Matches between treated and untreated    
    """
    idx_treat = treat == 1
    idx_rf = x == 1
    idx_treat_rf = np.where(idx_treat & idx_rf)[0]
    idx_treat_norf = np.where(idx_treat & ~idx_rf)[0]
    idx_notreat_rf = np.where(~idx_treat & idx_rf)[0]
    idx_notreat_norf  = np.where(~idx_treat & ~idx_rf)[0]
    
    # combine with risk factor
    n = np.min([len(idx_treat_rf), len(idx_notreat_rf)])
    if n > 0:
        idx_treat_rf = idx_treat_rf[np.random.choice(len(idx_treat_rf), n, replace=False)]
        idx_notreat_rf = idx_notreat_rf[np.random.choice(len(idx_notreat_rf), n, replace=False)]
    else:
        idx_treat_rf = np.array([],dtype=int)
        idx_notreat_rf = np.array([],dtype=int)
    
    # combine without risk factor
    n = np.min([len(idx_treat_norf), len(idx_notreat_norf)])
    if n > 0:
        idx_treat_norf = idx_treat_norf[np.random.choice(len(idx_treat_norf), n, replace=False)]
        idx_notreat_norf = idx_notreat_norf[np.random.choice(len(idx_notreat_norf), n, replace=False)]
    else:
        idx_treat_norf = np.array([],dtype=int)
        idx_notreat_norf = np.array([],dtype=int)
    
    idx = np.hstack([idx_treat_rf,idx_notreat_rf,idx_treat_norf,idx_notreat_norf])

    return idx
    

def _distance_match(pwd, pwd_max):    
    n = pwd.shape[0]
    m = pwd.shape[1]
    nmatched = 0
    mpairs = -1*np.ones((n,2),int)    
    for i in range(n):
        candidates = np.argsort(pwd[i,:])         
        for j in range(m):
            if candidates[j] not in mpairs[:,1]:
                break   
        if pwd[i,candidates[j]] < pwd_max:
            mpairs[nmatched,0] = i    
            mpairs[nmatched,1] = candidates[j]    
            nmatched=nmatched+1   
    return mpairs[:nmatched,]        

def match_dist(x, treat, caliper=0.5):
    """
    Produce Matches between treated and untreated using distances
    """
    from numpy.matlib import repmat
    bool_treat = treat == 1
    
    x1 = np.transpose(repmat(x[bool_treat],sum(~bool_treat),1))  
    x2 = repmat(x[~bool_treat],sum(bool_treat),1)
    dmat = np.abs(x1-x2)  # pairwise distance matrix
    
    mpairs = _distance_match(dmat, caliper)    
    idx_treat = np.where(bool_treat)[0][mpairs[:,0]]
    idx_control = np.where(~bool_treat)[0][mpairs[:,1]] 
    matches = np.hstack((idx_treat,idx_control))    
    return matches
    
    

def scenario_A(N = 200,
               n_iter = 5000,
               p_has_risk_factor = 0.1,
               p_treat_given_risk_factor = 0.95,
               p_treat_given_no_risk_factor = 0.05,
               p_outcome_given_risk_factor = 0.8,
               p_outcome_given_no_risk_factor = 0.1,
               method = None):
    """
    Scenario A. Binary risk factor.
    """        
    
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2')  
    
    estimated_treatment_effect = np.zeros((n_iter,))
    
    nvalid = 0  # number of "valid" runs
    for itera in range(n_iter):
        x = np.random.binomial(1,p_has_risk_factor,size=(N,1))
        treat = np.random.binomial(1,p_treat_given_risk_factor*x+p_treat_given_no_risk_factor*(1-x))         
        y = np.random.binomial(1,p_outcome_given_risk_factor*x+p_outcome_given_no_risk_factor*(1-x))    
        X = np.hstack([x, treat])   
        if method == 'matchexact':
            idx = scenario_A_match_exact(x, treat)            
            X = X[idx,:]
            y = y[idx]    
        elif method == 'matchdist':            
            idx = match_dist(x, treat)              
            X = X[idx,:]
            y = y[idx]            
        try:
            clf.fit(X,np.reshape(y, (len(y),)))
            estimated_treatment_effect[itera] = clf.coef_[0][1]
            nvalid = nvalid+1
        except ValueError:            
            estimated_treatment_effect[itera] = 0.        
    cis = [np.percentile(estimated_treatment_effect,2.5), np.percentile(estimated_treatment_effect,97.5)]     
    return cis, nvalid
       
##############################################################################
# Scenario B. Continuous risk factor, offset by distribution location between control/treated
##############################################################################
    
def scenario_B(N = 100,
               n_iter = 1000,
               loc = 1.,
               distribution_type = 'uniform',
               method = None,
               **kwargs):
    """
    Scenario B. Continuous risk factor; control/treated offset by distribution location 
    """        
        
    from sklearn.linear_model import LogisticRegression
    logisticfun = lambda t: 0.5+0.5*np.tanh(t)
    clf = LogisticRegression(penalty='l2')  
    
    if 'caliper' in kwargs:
        caliper = kwargs['caliper']
    else:
        caliper = 0.2 
    
    estimated_treatment_effect = np.zeros((n_iter,)) 
    treat = np.vstack([np.zeros((N,1)),np.ones((N,1))])
    nvalid = 0  # number of "valid" runs
    for itera in range(n_iter):   
        # generate the risk factor   
        if distribution_type == 'gaussian':        
            x = np.vstack([np.random.normal(loc=0.,scale=1.,size=(N,1)), np.random.normal(loc=loc,scale=1.,size=(N,1))])      
        elif distribution_type == 'uniform':
            x = np.vstack([np.random.uniform(low=0.,high=1.,size=(N,1)), np.random.uniform(low=loc,high=loc+1.,size=(N,1))])  
        else:
            raise ValueError('distribution_type [{}] not defined.'.format(distribution_type)) 
        X = np.hstack([x, treat])
        y = np.random.binomial(1,logisticfun(X[:,0])) # generate the outcome
        if method == 'matchdist':            
            idx = match_dist(x, treat, caliper)              
            X = X[idx,:]
            y = y[idx]
        try:
            clf.fit(X,np.reshape(y, (len(y),)))
            estimated_treatment_effect[itera] = clf.coef_[0][1]
            nvalid = nvalid+1
        except ValueError:            
            estimated_treatment_effect[itera] = 0.        
    cis = [np.percentile(estimated_treatment_effect,2.5), np.percentile(estimated_treatment_effect,97.5)]     
    return cis, nvalid
  
    
##############################################################################
# Scenario C. Continuous risk factor, treatment correlated to risk factor.
##############################################################################
    
def scenario_C(N = 100,
               n_iter = 1000,
               scale = 1.,
               distribution_type = 'uniform',
               method = None,
               **kwargs):
    """
    Scenario C. Continuous risk factor, treatment correlated to risk factor.
    Strength of correlation specified by scale_    
    
    n_iter = number of iterations for generating CIs
    numsca = number of scales (ie. standard deviations to try)
    N = number of samples per distribution    
    x = the risk factor
    treat = the treatment value
    """    
 
    from sklearn.linear_model import LogisticRegression
    logisticfun = lambda t: 0.5+0.5*np.tanh(t)
    clf = LogisticRegression(penalty='l2')
    
    if 'caliper' in kwargs:
        caliper = kwargs['caliper']
    else:
        caliper = 0.2 
       
    estimated_treatment_effect = np.zeros((n_iter,)) 
    nvalid = 0  # number of "valid" runs     
    for itera in range(n_iter):
        # generate the risk factor
        if distribution_type == 'singlegaussian' or distribution_type == 'gaussian':        
            x = np.vstack([np.random.normal(loc=0.,scale=1.,size=(N,1))]) 
        elif distribution_type == 'doublegaussian':
            x = np.vstack([np.random.normal(loc=0.,scale=1.,size=(N,1)), np.random.normal(loc=1.,scale=1.,size=(N,1))])   
        elif distribution_type == 'uniform':
            x = np.vstack([np.random.uniform(low=0.,high=1.,size=(N,1))])   
        else:
            raise ValueError('distribution_type [{}] not defined.'.format(distribution_type))            
        treat = np.reshape([np.random.normal(loc=a,scale=scale) for a in x], newshape=x.shape)
        X = np.hstack([x, treat])
        y = np.random.binomial(1,logisticfun(X[:,0])) # generate the outcome
        if method == 'matchdist':            
            idx = match_dist(x, treat, caliper)              
            X = X[idx,:]
            y = y[idx]
        try:
            clf.fit(X,np.reshape(y, (len(y),)))
            estimated_treatment_effect[itera] = clf.coef_[0][1]
            nvalid = nvalid+1
        except ValueError:            
            estimated_treatment_effect[itera] = 0.        
    cis = [np.percentile(estimated_treatment_effect,2.5), np.percentile(estimated_treatment_effect,97.5)]     
    return cis, nvalid
    



##############################################################################
# MAIN
##############################################################################


def main():

    print('Running Scenario A...')    
    
    N = 200
    p_has_risk_factor = 0.25
    p_treat_given_risk_factor = 0.90
    p_treat_given_no_risk_factor = 0.10
    p_outcome_given_risk_factor = 0.90
    p_outcome_given_no_risk_factor = 0.10
    n_iter = 1000    
    method = 'regression'   
    cis, _ = scenario_A(N,
                        n_iter,
                        p_has_risk_factor,
                        p_treat_given_risk_factor,
                        p_treat_given_no_risk_factor,
                        p_outcome_given_risk_factor,
                        p_outcome_given_no_risk_factor,
                        method)
                        
    return print('Treatment Effect CI: ({:.2f},  {:.2f})'.format(cis[0],cis[1]))

##############################################################################
    

if __name__ == "__main__":
    main()