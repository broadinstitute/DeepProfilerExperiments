import numpy as np
import scipy.linalg
import pandas as pd

class WhiteningNormalizer(object):
    def __init__(self, controls, reg_param=1e-6):
        # Whitening transform on population level data
        self.mu = controls.mean()
        self.whitening_transform(controls - self.mu, reg_param, rotate=True)
        print(self.mu.shape, self.W.shape)
        
    def whitening_transform(self, X, lambda_, rotate=True):
        C = (1/X.shape[0]) * np.dot(X.T, X)
        s, V = scipy.linalg.eigh(C)
        D = np.diag( 1. / np.sqrt(s + lambda_) )
        W = np.dot(V, D)
        if rotate:
            W = np.dot(W, V.T)
        self.W = W

    def normalize(self, X):
        return np.dot(X - self.mu, self.W)
    

def load_similarity_matrix(filename):
    # Load matrix in triplet format and reshape
    cr_mat = pd.read_csv(filename)
    X = cr_mat.pivot(index="Var1", columns="Var2", values="value").reset_index()
    
    # Identify annotations
    Y = cr_mat.groupby("Var1").max().reset_index()
    Y = Y[~Y["Metadata_moa.x"].isna()].sort_values(by="Var1")
    
    # Make sure the matrix is sorted by treatment
    X = X.loc[X.Var1.isin(Y.Var1), ["Var1"] + list(Y.Var1)].sort_values("Var1")
    
    return X,Y
    