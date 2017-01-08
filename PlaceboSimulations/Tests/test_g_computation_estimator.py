# -*- coding: utf-8 -*-
"""
Test the G-computation estimator for the treatment effect.
"""

from __future__ import print_function
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

"""
Scenario A. Binary risk factor.
"""        

N = 200
p_has_risk_factor = 0.25
p_treat_given_risk_factor = 0.90
p_treat_given_no_risk_factor = 0.10
p_outcome_given_risk_factor = 0.90
p_outcome_given_no_risk_factor = 0.10
n_iter = 10000

##############################################################################

def treatment_effect_estimator(X,clf,method='gcomp'):
    """
    Estimate treatment effect; default: G-computation estimator
    
    Assumes binary treatment coded 0/1;
    binary outcome coded 0/1
    1-dimensional covariate
    
    """    
    if method == 'gcomp':
        X_ = lambda t : np.hstack([X[:,[0]], t*np.ones((X.shape[0],1))])
        if hasattr(clf, 'predict_proba'):            
            Q = lambda t : clf.predict_proba(X_(t))[:,clf.classes_==1]
        elif hasattr(clf, 'predict'):
            # catch linear regression case and ??
            Q = lambda t : clf.predict(X_(t))
        else:
            raise ValueError('clf [{}] has unrecognized predict method.'.format(clf))                
        treatment_effect = np.mean(Q(1) - Q(0)) 
    elif method == 'coef':
        # Note this is the wrong way to esimate treatment effect
        if hasattr(clf, 'predict_proba'):            
            treatment_effect = clf.coef_[0][1]
        elif hasattr(clf, 'predict'):
            # catch linear regression case and ??
            treatment_effect = clf.coef_[1]
        else:
            raise ValueError('clf [{}] has unrecognized coef_ method.'.format(clf))
    else:
        raise ValueError('method [{}] not defined.'.format(method)) 

    return treatment_effect

##############################################################################



clf = LogisticRegression(penalty='l2')
clf2 = LinearRegression()

estimator_desc = ['gcomp, logistic, correct spec raw', 
                  'gcomp, logistic, correct spec fun',
                  'coef, logistic, correct spec raw, invalid estimator',
                  'coef, logistic, correct spec fun, invalid estimator',
                  'gcomp, linear, incorrect spec fun',
                  'coef, linear, incorrect spec fun',
                  'placeholder']
estimated_treatment_effect = np.zeros((n_iter,7))


nvalid = 0  # number of "valid" runs
for itera in range(n_iter):
    x = np.random.binomial(1,p_has_risk_factor,size=(N,1))
    treat = np.random.binomial(1,p_treat_given_risk_factor*x+p_treat_given_no_risk_factor*(1-x))         
    y = np.random.binomial(1,p_outcome_given_risk_factor*x+p_outcome_given_no_risk_factor*(1-x))    
    X = np.hstack([x, treat])      
    try:
        clf.fit(X,np.reshape(y, (len(y),)))
        X_ = lambda t : np.hstack([X[:,[0]], t*np.ones((X.shape[0],1))])
        q = lambda t : clf.predict_proba(X_(t))[:,clf.classes_==1]
        estimated_treatment_effect[itera,0] = np.mean(q(1) - q(0))
        estimated_treatment_effect[itera,1] = treatment_effect_estimator(X,clf,method='gcomp')
        estimated_treatment_effect[itera,2] = clf.coef_[0][1]
        estimated_treatment_effect[itera,3] = treatment_effect_estimator(X,clf,method='coef')
        # Try linear model
        clf2.fit(X,np.reshape(y, (len(y),)))
        estimated_treatment_effect[itera,4] = treatment_effect_estimator(X,clf2,method='gcomp')
        estimated_treatment_effect[itera,5] = treatment_effect_estimator(X,clf2,method='coef')
        nvalid = nvalid+1
    except ValueError:            
        estimated_treatment_effect[itera] = 0. 

for s, cil, ciu in zip(estimator_desc,
                       np.percentile(estimated_treatment_effect,2.5,axis=0),
                       np.percentile(estimated_treatment_effect,97.5,axis=0)):
    print(cil, ciu, s)
