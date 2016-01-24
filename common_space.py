import os
import numpy as np
import sklearn.metrics.pairwise as skpw
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from utils import plot_similarity_matrix
os.sys.path.append('./kernel_two_sample_test')
from kernel_two_sample_test.kernel_two_sample_test import MMD2u, compute_null_distribution


def common_space_representation(data_b6, data_btbr):
    """
    Computes the representation in the common space. This function can be
    used by both the structural and functional data.
    """
    X = np.vstack((data_b6, data_btbr))
    #Computing the empirical cumulative distribution function using all 
    #non-zero edge weights.
    ecdf = ECDF(X.reshape(-1)[X.reshape(-1) > 0])
    norm_b6 = np.zeros(data_b6.shape)
    norm_btbr = np.zeros(data_btbr.shape)
    #Representing each edge weight by its distribution value.
    for i in range(len(data_b6)):
        norm_b6[i] = ecdf(data_b6[i])
    for i in range(len(data_btbr)):
        norm_btbr[i] = ecdf(data_btbr[i])
        
    return norm_b6, norm_btbr
       
    
    
def compute_kernel_matrix(struc_b6, struc_btbr, func_b6, func_btbr, 
                          kernel='linear', normalized=True, plot=True, **kwds):
    """
    Computes the kernel matrix for all graphs (structural and functional)
    represented in the common space.
    
    Parameters:
    ----------
    struc_b6: array like
    struc_btbr: array like
    func_b6: array like
    func_btbr: array like
    kernel: string
            Kernel measure. The kernels implemented in sklearn are allowed.
            Possible values are 'rbf', 'sigmoid', 'polynomial', 
            'poly', 'linear', 'cosine'.
    normalized: boolean
                Whether to normalize the kernel values by
                k_normalized(a,b) = k(a,b)/np.sqrt(k(a,a)*k(b,b))
    **kwds: optional keyword parameters
            Any further parameters are passed directly to the kernel function.
    Returns:
    ------
    k_mat: ndarray
           Kernel matrix
    """
    vects = np.vstack((struc_b6, struc_btbr, func_b6, func_btbr))    
    k_mat = skpw.pairwise_kernels(vects, vects, metric = kernel, **kwds)
    if normalized:
        k_norm = np.zeros(k_mat.shape)
        for i in range(len(k_mat)):
            for j in range(i, len(k_mat)):
                k_norm[i, j] = k_norm[j, i] = k_mat[i, j] / np.sqrt(k_mat[i, i]
                * k_mat[j, j])   
        k_mat = k_norm
        
    if plot:
         plot_similarity_matrix(k_mat)
    
    return k_mat
    

def compute_mmd_struc_func(k_mat, struc_b6, struc_btbr, func_b6, func_btbr, 
                           iterations=100000, plot=True):
    """
    Computes the mmd values for the structural and functional problems and plot
    them with the null distributions.
    
    Parameters:
    ----------
    k_mat: ndarray
           Kernel matrix
    struc_b6: array like
           Structural vectors for B6 class
    struc_btbr: array like
           Structural vectors for BTBR class
    func_b6: array like
           Functional vectors for B6 class
    func_btbr: array like
           Functional vectors for BTBR class
    """
    #Computing the number of samples belonging to structural data in order
    #to split the kernel matrix.
    l_struc = len(struc_b6) + len(struc_btbr)
    
    #Computing MMD values
    struc_mmd = MMD2u(k_mat[:l_struc][:,:l_struc], len(struc_b6), 
                      len(struc_btbr))
    func_mmd = MMD2u(k_mat[l_struc:][:,l_struc:], len(func_b6), len(func_btbr))
    print "struc_mmd = %s, func_mmd = %s" %(struc_mmd, func_mmd) 
    
    #Computing the null-distribution
    mmd2u_null_all = compute_null_distribution(k_mat, 
                                               struc_b6.shape[0]+func_b6.shape[0], 
                                               struc_btbr.shape[0]+func_btbr.shape[0], 
                                               iterations, seed=123, 
                                               verbose=False)
    #Computing the p-value
    struc_p_value = max(1.0/iterations, 
                        (mmd2u_null_all > struc_mmd).sum() / float(iterations))
    print("struc_p-value ~= %s \t (resolution : %s)" % (struc_p_value, 
                                                        1.0/iterations))
    func_p_value = max(1.0/iterations, 
                       (mmd2u_null_all > func_mmd).sum() / float(iterations))
    print("func_p-value ~= %s \t (resolution : %s)" % (func_p_value, 
                                                       1.0/iterations))
                                                       
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        prob, bins, patches = plt.hist(mmd2u_null_all, bins=50, normed=True)
        ax.plot(struc_mmd, prob.max()/30, 'w*', markersize=15, 
                markeredgecolor='k', markeredgewidth=2, 
                label="$MMD^2_S = %s$" % struc_mmd)
        ax.plot(func_mmd, prob.max()/30, 'w^', markersize=15,
                markeredgecolor='k', markeredgewidth=2, 
                label="$MMD^2_F = %s$" % func_mmd)
        plt.xlabel('$MMD^2_u$')
        plt.ylabel('$p(MMD^2_u)$')
#        plt.title('$MMD^2_u$: null-distribution and observed values')
        
        ax.annotate('p-value: %s' %(struc_p_value), xy=(float(struc_mmd), 4.),  
                    xycoords='data',
                    xytext=(-105, 30), textcoords='offset points',
                    bbox=dict(boxstyle="round", fc="1."),
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                    )
                        
        ax.annotate('p-value: %s' %(func_p_value), xy=(float(func_mmd), 4.), 
                    xycoords='data',
                    xytext=(10, 30), textcoords='offset points',
                    bbox=dict(boxstyle="round", fc="1."),
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                    )
                        
        plt.legend(numpoints=1)
    
    
    
def compute_distance_mmd(k_mat, struc_b6, struc_btbr, func_b6, func_btbr, 
                         iterations=100000, plot=True):
    """
    Computes the distance of structural and functional mmd values and compares
    it with the null distribution.
    
    Parameters:
    ----------
    k_mat: ndarray
           Kernel matrix
    struc_b6: array like
           Structural vectors for B6 class
    struc_btbr: array like
           Structural vectors for BTBR class
    func_b6: array like
           Functional vectors for B6 class
    func_btbr: array like
           Functional vectors for BTBR class
    """
    #Computing the number of samples belonging to structural data in order
    #to split the kernel matrix.
    l_struc = len(struc_b6) + len(struc_btbr)
    
    #Computing dist mmd
    struc_mmd = MMD2u(k_mat[:l_struc][:,:l_struc], len(struc_b6), 
                      len(struc_btbr))
    func_mmd = MMD2u(k_mat[l_struc:][:,l_struc:], len(func_b6), 
                     len(func_btbr))
    dist_mmd = struc_mmd - func_mmd
    
    #Computing null distribution
    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        idx = np.random.permutation(len(k_mat))
        k_perm = k_mat[idx][:,idx]
        s_mmd = MMD2u(k_perm[:l_struc][:,:l_struc], len(struc_b6),
                      len(struc_btbr))
        f_mmd = MMD2u(k_perm[l_struc:][:,l_struc:], len(func_b6), 
                      len(func_btbr))
        mmd2u_null[i] = s_mmd - f_mmd
    
    #Computing p-value
    dist_p_value = max(1.0/iterations, 
                       (mmd2u_null > dist_mmd).sum() / float(iterations))
    print "Dist p-value ~= %s \t (resolution : %s)" % (dist_p_value, 
                                                       1.0/iterations)
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        prob, bins, patches = plt.hist(mmd2u_null, bins=50, normed=True)
        ax.plot(dist_mmd, prob.max()/30, 'w*', markersize=15, markeredgecolor='k',
                markeredgewidth=2, label="$MMD^2_{SF} = %s$" % dist_mmd)
    
        plt.xlabel('$MMD^2_u$')
        plt.ylabel('$p(MMD^2_u)$')
        plt.legend(numpoints=1)
        
        ax.annotate('p-value: %s' %(dist_p_value), xy=(float(dist_mmd), 1.), 
                xycoords='data',
                xytext=(10, 30), textcoords='offset points',
                bbox=dict(boxstyle="round", fc="1."),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                )        
   
        
  
    
    
    
    
    
    
    