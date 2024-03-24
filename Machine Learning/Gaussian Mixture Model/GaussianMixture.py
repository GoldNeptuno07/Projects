# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:19:13 2024

@author: GoldNeptuno07
"""
# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from typing import List

class GaussianMixture:
    """
        A class to performe the GaussianMixture model, performing 
        multivariate normal distribution. 
    """
    def __init__(self, n_components:int):
        """
            Args: 
                n_components. An integer value representing the number of clusters.
        """
        self.k= n_components
        self.gamma : List[List[float]]
        self.mean : List[List[float]]
        self.cov : List[List[float]]
        self.pi : List[List[float]]
    
    def _init_params_(self, X: List[List[float]]):
        """
            We will performe KMeans to find the optimal means for each cluster. Then initialize the parameters, such as, 
            the mean, the covarianze matrix and the pi values(probability of each cluster).
        """
        n_samples, n_features = X.shape
        self.mean = KMeans(n_clusters= self.k, n_init='auto').fit(X).cluster_centers_
        self.cov = np.array([np.eye(n_features) for i in range(self.k)])
        self.pi = np.full([1, self.k], 1/self.k)
        
    def ExpectationStep(self, X: List[List[float]]):
        """
            Compute the Expectation Step calculing the gamma of each sample for each cluster.
        """
        n_samples, n_features = X.shape
        p_xz = np.zeros([n_samples, self.k])
        for k, (mean, cov) in enumerate(zip(self.mean, self.cov)):
            p_xz[:, k] = multivariate_normal(mean, cov).pdf(X)
        p_x = p_xz * self.pi
        self.gamma = p_x / np.sum(p_x, axis= 1, keepdims= True)
    
    def MaximizationStep(self, X: List[List[float]]):
        """
            Compute the Maximization Step, optimizing the parameters based on the gammas.
            
            Args:
                X. A list containing the samples.
        """
        n_samples, n_features = X.shape
        mean_vect= []
        cov_vect = []
        pi_vect = []
        for k, (gamma, mean) in enumerate(zip(self.gamma.T, self.mean)): 
            Nk= np.sum(gamma)
            gamma_2d= gamma.reshape(-1, 1)
            # Compute the new mean
            new_mean= np.sum(gamma_2d * X, axis= 0) / Nk
            # Compute the new covarianze matrix
            X_mean = X - mean
            dot_product= np.array([np.outer(x,x) for x in X_mean])
            new_cov= np.sum(dot_product * gamma_2d.reshape(-1,1,1), axis= 0) / Nk
            # Compute the new pi values
            new_pi= np.mean(gamma)
            # Save new parameters
            self.mean[k]= new_mean
            self.cov[k]= new_cov
            self.pi[0, k]= new_pi
    
    
    def fit(self, X, max_iter= 100):
        """
            Method to train the Gaussian Mixture Model.
            
            Args.
                X. A list containing the samples.
                max_iter. An integer number. Maximum number of iterations. default= 100
            
        """
        # Array to Numpy Array
        X= np.array(X, dtype= np.float32)
        # Initialize Paramters
        self._init_params_(X)
        # Main Lopp
        for ite in range(1, max_iter+1):
            self.ExpectationStep(X)
            self.MaximizationStep(X)
        return self
            
    def predict(self, X: List[List[float]]):
        """
            Method to return the assigned cluster to each sample.
            return List[int]
        """
        n_samples, n_features = X.shape
        p_xz = np.zeros([n_samples, self.k])
        for k, (mean, cov) in enumerate(zip(self.mean, self.cov)):
            p_xz[:, k] = multivariate_normal(mean, cov).pdf(X)
        return np.argmax(p_xz, axis= 1)
        
    def predict_proba(self, X: List[List[float]]):
        """
            Method to return the probability of each cluster.
            return List[List[float]]
        """
        n_samples, n_features = X.shape
        p_xz = np.zeros([n_samples, self.k])
        for k, (mean, cov) in enumerate(zip(self.mean, self.cov)):
            p_xz[:, k] = multivariate_normal(mean, cov).pdf(X)
        return p_xz
            
            
            
            
            
            
            
            
            
            
            
            
            
            