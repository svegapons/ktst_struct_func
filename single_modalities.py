import os
import numpy as np
from sklearn.metrics import pairwise_kernels, pairwise_distances
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from utils import plot_similarity_matrix
os.sys.path.append('./kernel_two_sample_test')
from kernel_two_sample_test.kernel_two_sample_test import MMD2u, compute_null_distribution
import matplotlib.pyplot as plt

def MMD_single_modality(data_b6, data_btbr, modality='Structural',
                             iterations=100000, plot=True):
    """
    Process the data with the following approach: Embedding + 
    RBF_kernel + KTST
    Parameters:
    -----------
    
    Return:
    ----------
        MMD distance, null_distribution, p-value
    """
    print 'Analyzing %s data' %(modality)
    
    #Concatenating the data
    vectors = np.vstack((data_b6, data_btbr))
    n_b6 = len(data_b6)
    n_btbr = len(data_btbr)
   
    sigma2 = np.median(pairwise_distances(vectors, metric='euclidean'))**2    
    k_matrix = pairwise_kernels(vectors, metric='rbf', gamma=1.0/sigma2)    
    
    if plot:
        plot_similarity_matrix(k_matrix)
    
    #Computing the MMD
    mmd2u = MMD2u(k_matrix, n_b6, n_btbr)
    print("MMD^2_u = %s" % mmd2u)    
    #Computing the null-distribution
        
    #Null distribution only on B6 mice
#    sigma2_b6 = np.median(pairwise_distances(vectors_cl1, metric='euclidean'))**2    
#    k_matrix_b6 = pairwise_kernels(vectors_cl1, metric='rbf', gamma=1.0/sigma2_b6)
#    mmd2u_null = compute_null_distribution(k_matrix_b6, 5, 5, iterations, seed=123, verbose=False)
  
    mmd2u_null = compute_null_distribution(k_matrix, n_b6, n_btbr, iterations, 
                                           seed=123, verbose=False)
    
    print np.max(mmd2u_null)
    #Computing the p-value
    p_value = max(1.0/iterations, (mmd2u_null > mmd2u).sum() / float(iterations))
    print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0/iterations))    
    print 'Number of stds from MMD^2_u to mean value of null distribution: %s' % ((mmd2u - np.mean(mmd2u_null))/np.std(mmd2u_null))
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        prob, bins, patches = plt.hist(mmd2u_null, bins=50, normed=True)
        ax.plot(mmd2u, prob.max()/30, 'w*', markersize=15, 
                markeredgecolor='k', markeredgewidth=2, 
                label="$%s MMD^2_u = %s$" % (modality, mmd2u))
    #    func_p_value = max(1.0/iterations, (functional_mmd[1] > functional_mmd[0]).sum() / float(iterations))

        ax.annotate('p-value: %s' %(p_value), 
                    xy=(float(mmd2u), prob.max()/9.),  xycoords='data',
                    xytext=(-105, 30), textcoords='offset points',
                    bbox=dict(boxstyle="round", fc="1."),
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                    )
        plt.xlabel('$MMD^2_u$')
        plt.ylabel('$p(MMD^2_u)$')
        plt.legend(numpoints=1)
#        plt.title('%s_DATA: $p$-value=%s' %(modality, p_value))
        print ''
       
    

def SVM_single_modality(data_b6, data_btbr, modality='Structural'):
    """
    """
    print 'Analyzing %s data' %(modality)
    vectors = np.vstack((data_b6, data_btbr))
    y = np.hstack((np.zeros(len(data_b6)), np.ones(len(data_btbr))))
    sigma2 = np.median(pairwise_distances(vectors, metric='euclidean'))**2    
    k_matrix = pairwise_kernels(vectors, metric='rbf', gamma=1.0/sigma2)  

    clf = SVC(kernel='precomputed')
    cv_scores = cross_val_score(clf, k_matrix, y, cv=StratifiedKFold(y, n_folds=len(y)/2))
    
    print 'Mean accuracy: %s, std: %s' %(np.mean(cv_scores), np.std(cv_scores))
    print 'All folds scores: %s' %(cv_scores)
    print ''
        



