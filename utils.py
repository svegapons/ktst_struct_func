import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def load_structural_data(path_b6, path_btbr):
    """
    Load the structural connectivity matrixes from the .mat file.
    Parameters:
    ----------
    path_b6: string
              Path to the .mat file with the data of B6 class
    path_btbr: string
              Path to the .mat file with the data of BTBR class
    Returns:
    ------
    norm_b6: ndarray
            Array with structural data belonging to B6 class
    norm_btbr: ndarray
            Array with structural data belonging to BTBR class
    """
    print "Loading structural connectivity matrixes"
    struc_b6 = []
    #Loading the 3D matrix
    mat = np.log(1 + loadmat(path_b6)['AssocMatrixC57'])
#    mat = loadmat(path_b6)['AssocMatrixC57']

    for i in range(mat.shape[-1]):
        struc_b6.append(mat[:,:,i].reshape(-1))    
    print 'Number of subjects in class B6: %s' %(len(struc_b6))
    struc_b6 = np.array(struc_b6)
    
    struc_btbr = []
    #Loading the 3D matrix
    mat = np.log(1 + loadmat(path_btbr)['AssocMatrixBTBR'])
    for i in range(mat.shape[-1]):
        struc_btbr.append(mat[:,:,i].reshape(-1))
    print 'Number of subjects in class BTBR: %s' %(len(struc_btbr))
    struc_btbr = np.array(struc_btbr)
        
    return struc_b6, struc_btbr
    
    
    
def load_functional_data(path_b6, path_btbr):
    """
    Load the functional connectivity matrixes from the .mat file.
    Parameters:
    ----------
    path_b6: string
              Path to the .mat file with the data of B6 class
    path_btbr: string
              Path to the .mat file with the data of BTBR class
    Returns:
    ------
    norm_b6: ndarray
            Array with functional data belonging to B6 class
    norm_btbr: ndarray
            Array with functional data belonging to BTBR class            
    """
    print "Loading functional connectivity matrixes"
    func_b6 = []
    #Unfolding and concatenating the data for all subjects
    for f in os.listdir(path_b6):
        mat = loadmat(os.path.join(path_b6, f))['nw_re_arranged']
        func_b6.append(mat.reshape(-1))
    print 'Number of subjects in class B6: %s' %(len(func_b6))
    func_b6 = np.array(func_b6)
    #Removing the edges with negative correlation values
    func_b6 = np.where(func_b6 > 0, func_b6, 0)
    #Using as edge weights the absolute values 
#    func_b6 = np.abs(func_b6)

    #Creating graphs for class 1
    func_btbr = []
    #Unfolding and concatenating the data for all subjects
    for f in os.listdir(path_btbr):
        mat = loadmat(os.path.join(path_btbr, f))['nw_re_arranged']
        func_btbr.append(mat.reshape(-1))
    print 'Number of subjects in class BTBR: %s' %(len(func_btbr))
    func_btbr = np.array(func_btbr)
    #Removing the edges with negative correlation values
    func_btbr = np.where(func_btbr > 0, func_btbr, 0)
    #Using as edge weights the absolute values 
#    func_btbr = np.abs(func_btbr)
           
    return func_b6, func_btbr
    

def plot_similarity_matrix(k_matrix):
    """
    Plot a similarity matrix...
    """
    plt.figure()
    iplot = plt.imshow(k_matrix, interpolation='none')
    iplot.set_cmap('spectral')
    plt.colorbar()
#    plt.title('Similarity matrix')
    plt.show()    
    


    
    
    
    
