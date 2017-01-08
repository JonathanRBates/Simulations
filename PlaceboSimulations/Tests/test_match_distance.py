# -*- coding: utf-8 -*-
"""
Test matching functions and their efficiency.
"""

import numpy as np
from numpy.matlib import repmat
from treatment_effect_under_placebo_tests import _distance_match


def do_repmat1(x,treat):
    bool_treat = treat == 1
    x1 = np.transpose(repmat(x[bool_treat],sum(~bool_treat),1))  
    x2 = repmat(x[~bool_treat],sum(bool_treat),1)    
    return x1, x2
    
def do_repmat2(x,treat):
    # bool_treat = np.squeeze(treat) == 1
    bool_treat = treat[:,0] == 1
    x1 = repmat(x[bool_treat],1,sum(~bool_treat))
    x2 = np.transpose(repmat(x[~bool_treat],1,sum(bool_treat)))    
    return x1, x2

def get_dmat1(x,treat):
    x1, x2 = do_repmat1(x,treat)
    return np.abs(x1-x2)
    
def get_dmat2(x,treat):
    x1, x2 = do_repmat2(x,treat)
    return np.abs(x1-x2)
    
def matching1(dmat):
    caliper = 0.7
    mpairs = _distance_match(dmat, caliper)
    return mpairs
    
def matching2(dmat):
    # TODO: this
    # mpairs = linear_assignment(None)
    # return mpairs
    return None
    
    
##############################################################################
# Test Equivalence
##############################################################################

N = 200
for itera in range(1000):            
        # Generate Data
        x = np.random.binomial(1,0.3,size=(N,1))
        treat = np.random.binomial(1,0.7,size=(N,1))
        x1, x2 = do_repmat1(x,treat)
        x1b, x2b = do_repmat2(x,treat)
        assert  (x1 == x1b).all(), "x1 construction methods are not equivalent"
        assert  (x2 == x2b).all(), "x2 construction methods are not equivalent"

##############################################################################
# Test efficiency
##############################################################################
 
import timeit 
setup = """
import numpy as np
from numpy.matlib import repmat
N=200
x = np.random.binomial(1,0.3,size=(N,1))
treat = np.random.binomial(1,0.7,size=(N,1))
"""

print min(timeit.Timer("""
bool_treat = treat == 1
x1 = np.transpose(repmat(x[bool_treat],sum(~bool_treat),1))  
x2 = repmat(x[~bool_treat],sum(bool_treat),1)    
""", setup=setup).repeat(7, 100))

print min(timeit.Timer("""
bool_treat = np.squeeze(treat) == 1
x1 = repmat(x[bool_treat],1,sum(~bool_treat))
x2 = np.transpose(repmat(x[~bool_treat],1,sum(bool_treat)))
""", setup=setup).repeat(7, 100))


##############################################################################
# Try scikit learn
##############################################################################

from sklearn.metrics.pairwise import pairwise_distances
def get_dmat3(x,treat):
    bool_treat = treat[:,0] == 1
    x1 = x[bool_treat]
    x2 = x[~bool_treat]   
    return pairwise_distances(x1, x2, n_jobs=1)


N = 200
for itera in range(1000):            
        # Generate Data
        x = np.random.binomial(1,0.3,size=(N,1))
        treat = np.random.binomial(1,0.7,size=(N,1))
        dmat = get_dmat1(x,treat)
        dmat3 = get_dmat3(x,treat)
        assert  (dmat == dmat3).all(), "dmat construction methods are not equivalent"

setup = """
import numpy as np
from numpy.matlib import repmat
from sklearn.metrics.pairwise import pairwise_distances

def get_dmat1(x,treat):
    bool_treat = treat == 1
    x1 = np.transpose(repmat(x[bool_treat],sum(~bool_treat),1))  
    x2 = repmat(x[~bool_treat],sum(bool_treat),1)    
    return np.abs(x1-x2)

def get_dmat3(x,treat):
    bool_treat = treat[:,0] == 1
    x1 = x[bool_treat]
    x2 = x[~bool_treat]   
    return pairwise_distances(x1, x2, n_jobs=1)

N=200
x = np.random.binomial(1,0.3,size=(N,1))
treat = np.random.binomial(1,0.7,size=(N,1))
"""

print min(timeit.Timer("""
get_dmat1(x,treat)  
""", setup=setup).repeat(7, 1000))

print min(timeit.Timer("""
get_dmat3(x,treat)
""", setup=setup).repeat(7, 1000))


##############################################################################
# Tests
##############################################################################


"""
N = 200

from accelerate import profiler

if False:
    p = profiler.Profile(signatures=False)
    p.enable()
    for itera in range(1000):            
        # Generate Data
        x = np.random.binomial(1,0.3,size=(N,1))
        treat = np.random.binomial(1,0.7,size=(N,1))
        dmat = get_dmat1(x,treat)
        dmat2 = get_dmat2(x,treat)
        assert  (dmat == dmat2).all(), "dmat construction methods are not equivalent"
    p.disable()
    p.print_stats()    

    




    
for itera in range(1000):            
    # Generate Data
    x = np.random.binomial(1,0.3,size=(N,1))
    treat = np.random.binomial(1,0.7,size=(N,1))
    dmat = get_dmat1(x,treat)
    dmat2 = get_dmat2(x,treat)
    assert  (dmat == dmat2).all(), "dmat construction methods are not equivalent"
    mpairs = matching1(dmat)
    mpairs2 = matching2(dmat)
    # assert mpairs == mpairs2, "matching algorithms are not equivalent"

"""