# -*- coding: utf-8 -*-
"""
Treatment Effect Tests (Pseudo/Placebo Treatments)

One binary treatment with no effect on outcome. One risk factor.

Scenario A. Binary risk factor.

Scenario B. Continuous risk factor, offset by distribution location between control/treated

Scenario C. Continuous risk factor, treatment correlated to risk factor.
"""

from __future__ import print_function
import numpy as np

##############################################################################
# G-computation estimator for treatment effect
##############################################################################
  
def treatment_effect_estimator(X,clf,method='gcomp'):
    """
    Estimate treatment effect; default: G-computation estimator
    
    Assumes binary treatment coded 0/1;
    binary outcome coded 0/1
    1-dimensional covariate
    
    """    
    if method == 'gcomp':
        X1 = np.hstack([X[:,[0]], np.ones((X.shape[0],1))])
        X0 = np.hstack([X[:,[0]], 0.*np.ones((X.shape[0],1))])        
        if hasattr(clf, 'predict_proba'):   
            treatment_effect = np.mean(clf.predict_proba(X1)[:,clf.classes_==1] -
                                       clf.predict_proba(X0)[:,clf.classes_==1])            
        elif hasattr(clf, 'predict'):
            # catch linear regression case and ??
            treatment_effect = np.mean(clf.predict(X1)[:,clf.classes_==1] -
                                       clf.predict(X0)[:,clf.classes_==1])            
        else:
            raise NotImplementedError('clf [{}] has unrecognized predict method.'.format(clf))       
        
    elif method == 'coef':
        # Note this is the wrong way to esimate treatment effect unless linear link
        # catch linear regression case and ??
        raise Warning('Assuming linear link function for this estimator')
        treatment_effect = clf.coef_[1]
    else:
        raise NotImplementedError('method [{}] not defined.'.format(method)) 

    return treatment_effect

        

##############################################################################
# Matching Functions
##############################################################################

def _distance_match_1_to_1(pwd, pwd_max):   
    """
    Given distance matrix pwd, match each row to the nearest column, 
    conditional on the distance being bounded by the value of pwd_max.    
    """
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
    Produce Matches between treated and control using distances
    
    Note: This code assumes 1-d x
    
    """
    from scipy.spatial.distance import cdist    
    bool_treat = treat == 1    
    # Old, 1-1 approach: 
    #   from sklearn.metrics.pairwise import pairwise_distances
    #   pwd = pairwise_distances(x[bool_treat].reshape(-1,1), x[~bool_treat].reshape(-1,1), n_jobs=-1)  
    #   mpairs = _distance_match_1_to_1(dmat, caliper)  
    # Alternative, "optimal" approach:
    #   from scipy.optimize import linear_sum_assignment
    #   row_ind, col_ind = linear_sum_assignment(pwd) 
    #
    # Note: adding reshape(-1,1) for case dim(x) = 1
    pwd = cdist(x[bool_treat].reshape(-1,1), x[~bool_treat].reshape(-1,1), metric='euclidean') 
    match_id = np.argmin(pwd,axis=1)
    mpairs = np.array([[i, j] for (i,j) in zip(np.array(range(pwd.shape[0])), match_id) if pwd[i,j] < caliper])    
    if len(mpairs) > 0:
        idx_treat = np.where(bool_treat)[0][mpairs[:,0]]
        idx_control = np.where(~bool_treat)[0][mpairs[:,1]] 
        matches = np.hstack((idx_treat,idx_control))   
    else:
        matches = None
    return matches

def match_exact(x, treat):
    """
    Produce Exact Matches between treated and control    
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


##############################################################################
# Scenario A. Binary risk factor.
##############################################################################
 

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
        # if itera % 1000 == 0:            
        #     print('.',end='')        
        x = np.random.binomial(1,p_has_risk_factor,size=(N,))
        treat = np.random.binomial(1,p_treat_given_risk_factor*x+p_treat_given_no_risk_factor*(1-x))         
        y = np.squeeze(np.random.binomial(1,p_outcome_given_risk_factor*x+p_outcome_given_no_risk_factor*(1-x)))
        X = np.vstack([x, treat]).T
        if method == 'matchexact':
            idx = match_exact(x, treat)            
            X = X[idx,:]
            y = y[idx]    
        elif method == 'matchdist':            
            idx = match_dist(x, treat)              
            X = X[idx,:]
            y = y[idx]            
        try:
            clf.fit(X,y)
            estimated_treatment_effect[itera] = treatment_effect_estimator(X,clf,method='gcomp')       
            nvalid = nvalid+1
        except ValueError:            
            estimated_treatment_effect[itera] = 0.        
    # print()    
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
               te_estimator = 'gcomp',
               **kwargs):
    """
    Scenario B. Continuous risk factor; control/treated offset by distribution location 
    """        
        
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
        # if itera % 1000 == 0:
        #    print('.',end='')
        # generate the risk factor   
        if distribution_type == 'gaussian': 
            x = np.hstack([np.random.normal(loc=0.,scale=1.,size=(N,)), np.random.normal(loc=loc,scale=1.,size=(N,))])    
        elif distribution_type == 'uniform':
            x = np.hstack([np.random.uniform(low=0.,high=1.,size=(N,)), np.random.uniform(low=loc,high=loc+1.,size=(N,))])   
        else:
            raise ValueError('distribution_type [{}] not defined.'.format(distribution_type)) 
        X = np.vstack([x, treat]).T
        y = generate_binomial_outcome(X[:,0]) # generate the outcome
        if method == 'matchdist':   
            idx = match_dist(x, treat, caliper)                       
            X = X[idx,:]
            y = y[idx]
        try:
            clf.fit(X,y)
            estimated_treatment_effect[itera] = treatment_effect_estimator(X,clf,method=te_estimator)
            nvalid = nvalid+1
        except ValueError:            
            estimated_treatment_effect[itera] = 0.      
    # print()   
    cis = [np.percentile(estimated_treatment_effect,2.5), np.percentile(estimated_treatment_effect,97.5)]     
    return cis, nvalid
  
    
##############################################################################
# Scenario C. Continuous risk factor, treatment correlated to risk factor.
##############################################################################
    
def scenario_C(N = 100,
               n_iter = 1000,
               covariance = 1.,
               distribution_type = 'uniform',
               method = None,
               **kwargs):
    """
    Scenario C. Continuous risk factor, treatment correlated to risk factor.
    Strength of correlation specified by covariance_    
    
    n_iter = number of iterations for generating CIs
    numsca = number of covariances (ie. standard deviations to try)
    N = number of samples per distribution    
    x = the risk factor
    treat = the treatment value
    """    
 
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2')
    
    if 'caliper' in kwargs:
        caliper = kwargs['caliper']
    else:
        caliper = 0.2 
       
    estimated_treatment_effect = np.zeros((n_iter,)) 
    nvalid = 0  # number of "valid" runs     
    for itera in range(n_iter):
        # if itera % 100 == 0:
        #     print('.',end='')        
        # generate the risk factor
        if distribution_type == 'singlegaussian' or distribution_type == 'gaussian':        
            x = np.hstack([np.random.normal(loc=0.,scale=1.,size=(N,))]) 
        elif distribution_type == 'doublegaussian':
            x = np.hstack([np.random.normal(loc=0.,scale=1.,size=(N,)), np.random.normal(loc=1.,scale=1.,size=(N,))])   
        elif distribution_type == 'uniform':
            x = np.hstack([np.random.uniform(low=0.,high=1.,size=(N,))])   
        else:
            raise ValueError('distribution_type [{}] not defined.'.format(distribution_type))            
        treat = np.reshape([np.random.normal(loc=a,scale=covariance) for a in x], newshape=x.shape)
        X = np.vstack([x, treat]).T
        y = generate_binomial_outcome(X[:,0]) # generate the outcome
        if method == 'matchdist':            
            idx = match_dist(x, treat, caliper)              
            X = X[idx,:]
            y = y[idx]
        try:
            clf.fit(X,y)
            estimated_treatment_effect[itera] = treatment_effect_estimator(X,clf,method='gcomp')
            nvalid = nvalid+1
        except ValueError:            
            estimated_treatment_effect[itera] = 0.  
    # print()      
    cis = [np.percentile(estimated_treatment_effect,2.5), np.percentile(estimated_treatment_effect,97.5)]     
    return cis, nvalid
    
##############################################################################
# Helper Functions
##############################################################################

def generate_binomial_outcome(u):
    logisticfun = lambda t: 0.5+0.5*np.tanh(t)    
    return np.squeeze(np.random.binomial(1,logisticfun(u)))
    


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
    n_iter = 100
    method = 'regression'   
    cis, _ = scenario_A(N,
                        n_iter,
                        p_has_risk_factor,
                        p_treat_given_risk_factor,
                        p_treat_given_no_risk_factor,
                        p_outcome_given_risk_factor,
                        p_outcome_given_no_risk_factor,
                        method)
                        
    return print('Treatment Effect CI: ({:.3f},  {:.3f})'.format(cis[0],cis[1]))

##############################################################################
    

if __name__ == "__main__":
    main()