import numpy as np

def baseline1(A,klist):
    '''
    sum A[j1][j1]+...A[jk][jk]
    s.t. A[j1][j1]>= ... >= A[jk][jk]

    #input
    A: sample covariance matrix
    klist: list for sparsity parameter

    #output
    list for different k in baseline-1
    '''
    diagA = [A[i,i] for i in range(A.shape[0])]
    diagA.sort(reverse = True)
    ans = []
    for i in klist:
        ans.append(sum(diagA[j] for j in range(i))) 
    return ans



