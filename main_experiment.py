"""
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from utils import load_structural_data, load_functional_data
from single_modalities import MMD_single_modality, SVM_single_modality
from common_space import compute_distance_mmd, compute_mmd_struc_func 
from common_space import compute_kernel_matrix, common_space_representation

        
if __name__ == "__main__":
    
    np.random.seed(0)
    
    #Loading structural and functional connectivities from both classes 
    print '\n ### Loading the data ###'          
    path_struc_b6 = "./data/MatriciRoiLaterali/MatriciB6"
    path_struc_btbr = "./data/MatriciRoiLaterali/MatriciBTBR"
    path_func_b6 = "./data/interhemispherical_correlation_matrices_50_regions-1/WT_inter_hemispherical_correlation_matrices"
    path_func_btbr = "./data/interhemispherical_correlation_matrices_50_regions-1/BTBR_inter_hemispherical_correlation_matrices"

    #Number of iterations for the computation of null-distributions
    iters = 100000
    
    #Working independently with structural and functional data
    struc_b6, struc_btbr = load_structural_data(path_struc_b6, path_struc_btbr)
    func_b6, func_btbr = load_functional_data(path_func_b6, path_func_btbr)
   
    #Applying classifiers => k-fold cross-validation using SVM
    print '\n ### Leave-one-subject-out cross-validation ###'
    SVM_single_modality(struc_b6, struc_btbr, modality='Structural')    
    SVM_single_modality(func_b6, func_btbr, modality='Functional')
       
    #Applying KTST
    print '\n #### KTST ####'    
    MMD_single_modality(struc_b6, struc_btbr, modality='Structural', 
                        iterations=iters)
    MMD_single_modality(func_b6, func_btbr, modality='Functional', 
                        iterations=iters)   
    
    #Working in the common space
    print '\n ### Computing common space representations ###'
    struc_b6, struc_btbr = common_space_representation(struc_b6, struc_btbr)    
    func_b6, func_btbr = common_space_representation(func_b6, func_btbr)    

    #Computing the kernel matrix
    print '\n ### Analyzing Structural and Functional data in the common space ###'
    sigma2 = np.median(pairwise_distances(np.vstack((struc_b6,
                                                     struc_btbr, func_b6, 
                                                     func_btbr)),
                                                     metric='euclidean'))**2 
    k_mat = compute_kernel_matrix(struc_b6, struc_btbr, func_b6, 
                                  func_btbr, kernel='rbf', 
                                  normalized=False, plot=True, gamma=1./sigma2)
    
    
    #Computing structural and functional mmd in the common space
    compute_mmd_struc_func(k_mat, struc_b6, struc_btbr, 
                           func_b6, func_btbr, iters)   
        
    #Computing dist_mmd
    compute_distance_mmd(k_mat, struc_b6, struc_btbr, 
                         func_b6, func_btbr, iters)   
    
    
    
    
    
    
    