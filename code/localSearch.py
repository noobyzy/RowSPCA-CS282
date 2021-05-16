"""
The local search method finds a lower (primal) bounds for SPCA
Author: yuzy
Last Modified: 2021/05/16
"""

import numpy as np
import random
import copy

def sqrtMat(A):
    Lam, U = np.linalg.eig(A)
    Lam = np.sqrt(Lam)

    LamMat = np.diag(Lam)
    return U@LamMat

def supportSetinit(d, k):
    '''
    random initialize the support set

    #input
    d: sample dimension
    k: sparsity parameter
    
    #output
    ret: random support set, length k
    '''
    ret = []
    while len(ret) < k:
        num = random.randint(0, d-1)
        if num not in ret:
            ret.append(num)
    return ret

def matSuppR(A, supp):
    '''
    return all rows of A in supp
    '''
    return A[supp, :]

def matSuppRC(A, suppR, suppC):
    '''
    return submat of A, all rows in suppR, all cols in suppC
    '''
    tempA = A[suppR, :]
    return tempA[:, suppC]

def remCandidate(supp, Ahalf, V):
    '''
    find the removing candidate
    '''
    reducedList = []
    i = 0

    tempMat = Ahalf[supp,:] - V @ np.transpose(V) @ Ahalf[supp,:]
    for ji in supp:
        reducedVal = np.linalg.norm(Ahalf[ji,:])**2
        reducedVal -= np.linalg.norm(tempMat, ord='fro')**2
        i+=1
        reducedList.append(reducedVal)
    minReducedVal = min(reducedList)
    joutid = reducedList.index(minReducedVal)
    jout = supp[joutid]
    return jout, joutid, minReducedVal 

def entCandidate(comp, supp, Ahalf, V, jout, joutid):
    '''
    find the entering candidate
    '''
    suppcopy = copy.deepcopy(supp)
    suppcopy.pop(joutid)

    reducedList = []
    i = 0

    tempMat = V @ np.transpose(V)
    for ji in comp:
        suppcopy.append(ji)
        reducedVal = np.linalg.norm(Ahalf[ji,:])**2
        reducedVal -= np.linalg.norm(Ahalf[suppcopy,:] - tempMat@Ahalf[suppcopy,:], ord='fro')**2
        i+=1
        reducedList.append(reducedVal)
        suppcopy.pop()
    maxReducedVal = max(reducedList)
    jinid = reducedList.index(maxReducedVal)
    jin = comp[jinid]
    return jin, jinid, maxReducedVal 

def LocalSearch(A, k, r, T, d):
    '''
    the main structure of local search method

    #input
    A: covariance matrix
    k: sparsity parameter
    r: # of eigenvalues
    T: # of maximum iterations
    d: sample dimension

    #output
    V: a feasible solution for SPCA
    '''

    Ahalf = sqrtMat(A)

    # initialize support set and complement set
    Supp = supportSetinit(d, k)
    Comp = []
    for i in range(d):
        if i not in Supp:
            Comp.append(i)

    # Lam: [k,] eigenval
    # U: [k, k] eigenvec
    Asupp = matSuppRC(A, Supp, Supp)
    Lam, U =  np.linalg.eig(Asupp)

    # [k, r] eigenvecs corresponding to  top r eigenvals
    V = U[:, [i for i in range(r)]]

    for t in range(T):
        print(t)
        jout, joutid, reducedValout = remCandidate(Supp, Ahalf, V)
        jin, jinid, reducedValin = entCandidate(Comp, Supp, Ahalf, V, jout, joutid)
        if reducedValin > reducedValout:
            Supp.pop(joutid)
            Supp.append(jin)
            Comp.pop(jinid)
            Comp.append(jout)

            Lam, U =  np.linalg.eig(matSuppRC(A, Supp, Supp))
            V = U[:, [i for i in range(r)]]
        else:
            return V
    return V


if __name__ == '__main__':
    X = np.load("../dataset/Reddit/Matrix_Reddit_1500_extension.npy")
    A = (1.0/X.shape[1])*X@np.transpose(X)
    print(LocalSearch(A, 20, 2, 500, X.shape[0]))