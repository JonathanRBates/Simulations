# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import random
import matplotlib.pyplot as plt


##############################################################################

class SimulationA(object):
    
    def __str__(self):
        return('''
        This class simulates a single dichotomous outcome
        and dichotomous treatment.  The probability for each outcome y1/y0
        is given by a logistic model with coefficients gamma1/gamma0.
        The propensity score for treatment is given by a logistic model 
        with coefficients beta.  Patient baseline data is generated
        from a centered multivariate normal distribution with covariance cov.  
        
        Number of covariates: {}
        Risk 1 coefficients: {}
        Risk 0 coefficients: {}
        Propensity score coefficients: {}
        '''.format(self.p,self.gamma1,self.gamma0,self.beta))
                  
    def __init__(self, p=None, cov=None, gamma1=None, gamma0=None, beta=None):
        self.p = p               # number of covariates
        self.cov = cov           # baseline covarianstructure       
        self.gamma1 = gamma1     # for risk model with treatment          
        self.gamma0 = gamma0     # risk model without treatment        
        self.beta = beta         # propensity score model

    def __sample_dichotomous(self, x, coef):
        # Bernoulli RV with logistic link function
        # 0th coefficient is assumed to be bias
        N = x.shape[0]
        xaug = np.hstack([np.ones((N,1)),x])
        proj = np.dot(xaug, coef)
        r = 1./(1.+np.exp(-proj))
        # label = np.random.binomial(n=np.ones(N,dtype=np.int32), p=r)
        label = np.random.binomial(n=1, p=r)   
        return label, r
        
#        def sample_poisson(x, z, coef):
#        # Poisson RV with exponential link function
#        # 0th coefficient is assumed to be bias
#        N = x.shape[0]
#        xaug = np.hstack([np.ones((N,1)),x,z])
#        proj = np.dot(xaug, coef)
#        r = np.exp(proj)
#        y = np.random.poisson(r)
#        return y, r   
    
    
    def __generate_covariance(self, idx_1, idx_0, idx_e):
        # idx_1, idx_0, idx_e are indicator vectors of length p
        # this function generates a covariance matrix with block structure
        # according to intersections of idx_1, idx_0, idx_e      
        # We assume the noise covariates are uncorrelated.
        cov = np.identity(self.p)   
        for i1 in [False, True]:
            for i0 in [False, True]:
                for ie in [False, True]:
                    if ~i1 & ~i0 & ~ie:
                        # noise covariates are uncorrelated
                        continue
                    b = (idx_1==i1) & (idx_0==i0) & (idx_e==ie)
                    k = np.sum(b)   # block size   
                    if k == 0:
                        continue
                    idx_block = np.outer(b,b)                     
                    A = random.rand(k,k)                    
                    cov[idx_block] = np.dot(A,A.transpose()).ravel()                      
        self.cov = cov            
        
    def generate_model(self, p=10, thresh=0.):
        if (self.gamma1 is not None) or (self.gamma0 is not None) or (self.beta is not None):
            print('returning from generate model...')
            return 1            
        if self.p is None:
            self.p = p            
        [idx_1, idx_0, idx_e] = np.random.normal(0,1,(3,p))>thresh
        self.__generate_covariance(idx_1, idx_0, idx_e)
        self.gamma1 = np.random.normal(0,1,p+1)     # for risk model with treatment          
        self.gamma0 = np.random.normal(0,1,p+1)     # risk model without treatment        
        self.beta = np.random.normal(0,1,p+1)         # propensity score model
        
    def generate_baseline_data(self, N):       
        # possible extensions: GMM, heterogeneous data types        
        # (cf. test_gmm.py for additional ideas/code.)      
        mean = np.zeros(self.p)
        if self.cov is None:
            self.cov = np.identity(self.p)   
        return np.random.multivariate_normal(mean, self.cov, N)    
            
    def generate_outcomes(self, x):            
        y1, risk1 = self.__sample_dichotomous(x, self.gamma1)         
        y0, risk0 = self.__sample_dichotomous(x, self.gamma0)          
        z, propensity = self.__sample_dichotomous(x, self.beta)     
        yobs = z*y1+(1-z)*y0  
        outcomes = pd.DataFrame(np.transpose(np.vstack([yobs,z,propensity,y0,risk0,y1,risk1])), 
                             columns=['y','z','propensity','y0','risk0','y1','risk1'])      
        return outcomes

    def to_dataframe(self, x, outcomes):
        df_x = pd.DataFrame(x, columns=['x'+str(i) for i in range(1,self.p+1)])        
        return pd.concat([outcomes, df_x], axis=1)
    
    def mc_risk_difference(self, niter=1000, nsamples=1000):
        # use Monte Carlo integration to get the empirical marginal risk difference        
        delta = np.zeros(niter) # the emrd
        for i in range(niter):
            x = self.generate_baseline_data(nsamples)
            _, risk1 = self.__sample_dichotomous(x, self.gamma1)         
            _, risk0 = self.__sample_dichotomous(x, self.gamma0)          
            # marginal risk difference
            delta[i] = np.mean(risk1)-np.mean(risk0)
        print('Estimated marginal risk difference is {}'.format(np.mean(delta)))
        plt.figure()
        plt.hist(delta)        
        plt.xlabel('emrd')
        plt.show()
        # return delta
        
    def report(self, df):
        # Input: outcomes dataframe 
        #        or dataframe created by to_dataframe(self, x, outcomes)
        # Print
        #   (0) cross-tabulation of outcome/treatment
        #   (1) unadjusted risk difference
        #   (2) overlap of propensity scores

        t = pd.crosstab(df['y'],df['z'])
        print(t)
        temp = t/t.sum(axis=0)
        print('The unadjusted risk difference is ', temp[1][1]-temp[0][1])

        idx = df['z']
        plt.figure()
        plt.hist(df.loc[idx==1,'propensity'].values,bins=np.linspace(0,1,16),histtype='stepfilled',normed=True,alpha=0.4,label='z=1')        
        plt.hist(df.loc[idx==0,'propensity'].values,bins=np.linspace(0,1,16),histtype='stepfilled',normed=True,alpha=0.4,label='z=0')        
        plt.xlabel('true propensity score')
        plt.legend()
        plt.show()        
        
    def covariate_report(self, df, idx):
        # idx = covariate index of interest
        col = 'x'+str(idx)        
        fig = plt.figure()
        fig.add_subplot(121)
        idx = df['z']==1
        plt.hist(df.loc[idx,col].values,bins=10,histtype='stepfilled',normed=True,alpha=0.4,label='z=1')        
        plt.hist(df.loc[~idx,col].values,bins=10,histtype='stepfilled',normed=True,alpha=0.4,label='z=0')        
        plt.xlabel(col)
        plt.legend()                          
        fig.add_subplot(122)
        idxy = df['y']==1
        plt.hist(df.loc[idx & idxy,col].values,bins=10,histtype='stepfilled',normed=True,alpha=0.4,label='y=1 | z=1')        
        plt.hist(df.loc[~idx & idxy,col].values,bins=10,histtype='stepfilled',normed=True,alpha=0.4,label='y=1 | z=0')        
        plt.xlabel(col)        
        plt.legend()
        plt.show()     
        
##############################################################################
  
class SimulationA01(SimulationA):
    def __str__(self):
        return('''
        This class simulates a single dichotomous outcome
        and dichotomous treatment.  The probability for each outcome y1/y0
        is given by a logistic model with coefficients gamma1/gamma0.
        The propensity score for treatment is given by a logistic model 
        with coefficients beta.  Patient baseline data is generated
        from a centered multivariate normal distribution with identity covariance.

        Model Specs:
            gamma1 = [gamma01,  r1*cos(eps1), r1*sin(eps1),            0 ]
            gamma0 = [gamma00,  r2*cos(eps2), r2*sin(eps2),            0 ]
            beta   = [  beta0,            0 , r3*cos(eps3),  r3*sin(eps3)]     
            
        This model is inspired by 
        Brookhart et al., “Variable Selection for Propensity Score Models”, Am J Epi 2006.
        '''.format(self.p,self.gamma1,self.gamma0,self.beta))
           
    def __init__(self, bias=np.array([0.,0.,0.]), angle=np.array([-0.1,0.1,0.1])*np.pi, r=np.array([1.,1.,1.])):           
        p = 3        
        gamma1 = np.array([bias[0], r[0]*np.cos(angle[0]), r[0]*np.sin(angle[0]), 0.])    # outcome model coefficients
        gamma0 = np.array([bias[1], r[1]*np.cos(angle[1]), r[1]*np.sin(angle[1]), 0.])    # outcome model coefficients
        beta = np.array([bias[2], 0., r[2]*np.cos(angle[2]), r[2]*np.sin(angle[2])])        # propensity model coefficients
        SimulationA.__init__(self,p,np.identity(p),gamma1,gamma0,beta)    
                
    def contour(self,ss=20,arrows=True):    
        lgrid=np.linspace(-5,5,ss)
        X1, X2 = np.meshgrid(lgrid, lgrid)
        u = np.transpose(np.vstack([X1.ravel(), X2.ravel(), np.zeros(len(X1.ravel()))]))
        v = self.generate_outcomes(u)
        h = v['risk1']
        #-v['risk0']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(7,7)
        handle = plt.contour(X1, X2, h.reshape((ss,ss)))
        plt.clabel(handle, inline=1, fontsize=10)
        if arrows:
            ax.arrow(0, 0, 2.*self.gamma1[1], 2.*self.gamma1[2], head_width=0.3, head_length=.5, fc='r', ec='r')
            ax.arrow(0, 0, 2.*self.gamma0[1], 2.*self.gamma0[2], head_width=0.3, head_length=.5, fc='g', ec='g')
            ax.arrow(0, 0, 2.*self.beta[1],   2.*self.beta[2], head_width=0.3, head_length=.5, fc='k', ec='k')
        # plt.contour(X1, X2, v['risk1'].reshape((ss,ss)))
        # plt.contour(X1, X2, v['risk0'].reshape((ss,ss)))
        # plt.contour(X1, X2, v['propensity'].reshape((ss,ss)))
        plt.axis('equal')
        plt.title('Risk Difference')
        return plt

    def plot(self,df):    
        ss = 30
        lgrid=np.linspace(-5,5,ss)
        X1, X2 = np.meshgrid(lgrid, lgrid)
        u = np.transpose(np.vstack([X1.ravel(), X2.ravel(), np.zeros(len(X1.ravel()))]))
        v = self.generate_outcomes(u)
        h = v['risk1']-v['risk0']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(7,7)
        for (i, a) in zip([0,1],['^','o']):
            for (j, b) in zip([0,1],['g','g']):
                idx = (df['z']==i) & (df['y']==j)                
                plt.plot(df.loc[idx,'x1'], df.loc[idx,'x2'], a, color=b)       
        handle = plt.contour(X1, X2, h.reshape((ss,ss)))
        plt.clabel(handle, inline=1, fontsize=10)        
        ax.arrow(0, 0, 2.*self.gamma1[1], 2.*self.gamma1[2], head_width=0.3, head_length=.5, fc='r', ec='r')
        ax.arrow(0, 0, 2.*self.gamma0[1], 2.*self.gamma0[2], head_width=0.3, head_length=.5, fc='g', ec='g')
        ax.arrow(0, 0, 2.*self.beta[1],   2.*self.beta[2], head_width=0.3, head_length=.5, fc='k', ec='k')       
        plt.axis('equal')
        plt.title('Risk Difference')
        return plt
        
##############################################################################

class SimulationB(object):
    # King-Nielson type model
    
    def __str__(self):
        return('''
        (King-Nielson Type Simulation) 
        This class simulates a single dichotomous outcome
        and dichotomous treatment.  The outcome model is
        y = logistic(-3 + 1*z + 1*x_1 + 0*x_2 + 0.3*x_3 + epsilon)
        The treated z=1 are uniformly drawn from [w,w+1]^2 x [0,1]^(p-2)
        and controls are uniformly drawn from [0,1]^p.  
        
        Number of covariates: {}     
        Treatment shift w: {}
        Control pool ratio r: {}        (i.e. treated:control = 1:r)
        '''.format(self.p,self.w,self.r))
                  
    def __init__(self, p=2, w=0.2, r=1.):
        self.p = p              # number of covariates
        self.w = w              # shift between treated/controls   
        self.r = r              # treated:control = 1:r

    def generate_baseline_data(self, N):      
        Ntreated = N/(1+self.r)        
        x = np.random.uniform(low=0,high=1,size=(N,self.p))        
        x[:Ntreated,0] = x[:Ntreated,0]+self.w
        x[:Ntreated,1] = x[:Ntreated,1]+self.w
        z = np.zeros(N)        
        z[:Ntreated] = 1.
        return x, z   
        
    def generate_outcomes(self, x, z):
        N = x.shape[0]
        coef = np.zeros(self.p+2)
        coef[0] = -3.             # bias
        coef[1] = 1.              # treatment effect    
        coef[2] = 1.  
        coef[3] = 0.
        coef[4] = .3                         
        xaug = np.hstack([np.ones((N,1)), z.reshape((N,1)), x])
        noise = np.random.normal(0,0.01,N)
        proj = np.dot(xaug, coef) + noise
        r = 1./(1.+np.exp(-proj))
        y = np.random.binomial(n=1, p=r)       
        return y
     
    def to_dataframe(self, x, z, y):    
        N = x.shape[0]
        df = pd.DataFrame(np.hstack([y.reshape((N,1)),z.reshape((N,1)),x]), 
                          columns=['y','z']+['x'+str(i) for i in range(1,self.p+1)])   
        return df
        
        
##############################################################################
        
        
class SimulationMixtureA(object):
    
    def __str__(self):
        return('''
        This class simulates a single dichotomous outcome
        and dichotomous treatment.  The probability for each outcome y1/y0
        is given by ...
        
        Number of covariates: {}        
        '''.format(self.p))
                  
    def __init__(self, p=None, n_clusters=None):
        self.p = p                      # number of covariates
        self.n_clusters = n_clusters    # number of clusters; ? = np.random.poisson(3)  
        # self.gamma1 = gamma1            # for risk model with treatment          
        # self.gamma0 = gamma0            # risk model without treatment        
        # self.beta = beta                # propensity score model        
        
    def generate_model(self, cluster_var=20, cluster_sym=10):
        # cluster_var: is the variance of the centers, i.e. how distinguishable they are...        
        # cluster_sym: if not using wishart to sample covariance, how symmetric the clusters are
    
        ####################################################################        
        # cluster centers
        self._mean = np.random.multivariate_normal(np.zeros(self.p), cluster_var*np.eye(self.p), self.n_clusters)    
        
        ####################################################################
        # cluster covariances
        wishart_cov = False
        
        if wishart_cov:
            from scipy.stats import wishart
            self._cov = wishart.rvs(df=self.p, scale=np.eye(self.p)/self.p, size=self.n_clusters)
        else:            
            self._cov = []
            for i in range(self.n_clusters):
                u = np.random.multivariate_normal(np.zeros(self.p), np.eye(self.p), cluster_sym*self.p)
                self._cov.append(np.cov(u.transpose()))
        
        # test covariance
        for cov in self._cov:
            flag_symmetric = np.all(cov == np.transpose(cov))
            if ~flag_symmetric:
                raise Exception('Generated covariance is not symmetric.')            
            evals, evecs = np.linalg.eig(cov)
            flag_posdef = np.min(evals) > 0.
            if ~flag_posdef:
                raise Exception('Generated covariance is not positive definite. Eigenvalues: {}'.format(evals))
             
        ####################################################################
        # mixing proportions
        _pvals = np.random.uniform(0,1,self.n_clusters)
        self._pvals = _pvals/np.sum(_pvals)        
        
        ####################################################################
        # risk model; cluster-based treatment effects
        self.gamma1 = np.random.uniform(0.1,0.9,self.n_clusters)     # for risk model with treatment       
        self.gamma0 = np.random.uniform(0.1,0.9,self.n_clusters)     # risk model without treatment        
                
        ####################################################################
        # propensity model; cluster based propensities
        self.beta = np.random.uniform(0.1,0.9,self.n_clusters)       # propensity score model
        
    def generate_data(self, N): 
        # sample units
        _cluster = np.random.multinomial(n=1, pvals=self._pvals, size=N)  # this is only used to generate cluster size
        cluster_size = np.sum(_cluster, axis=0)        
        x = []  # baseline covariates     
        y0 = []  # outcome covariates
        y1 = []
        z = []  # treatment assignment
        for i in range(self.n_clusters):  
            x.append(np.random.multivariate_normal(self._mean[i], self._cov[i], cluster_size[i])) 
            y0.append(np.random.binomial(n=1, p=self.gamma0[i], size=(cluster_size[i],1)))   
            y1.append(np.random.binomial(n=1, p=self.gamma1[i], size=(cluster_size[i],1))) 
            z.append(np.random.binomial(n=1, p=self.beta[i], size=(cluster_size[i],1))) 
                            
        x = np.vstack(x)  
        y0 = np.vstack(y0)    
        y1 = np.vstack(y1)    
        z = np.vstack(z)         
        
        y = z*y1+(1-z)*y0  
        
        v = np.hstack([y,z,x])        
        np.random.shuffle(v)    # shuffle the order
        
        y = v[:,0].reshape(N)
        z = v[:,1].reshape(N)
        x = v[:,2:]        
         
        return x, y, z
            
 
        