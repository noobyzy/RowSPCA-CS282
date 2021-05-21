import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math

def upperbound_vector(A,r,k,T=120,N=200):
    '''
    PLA with SOCP-relaxation
    In vector form, require less preparing time

    max sum Lambda sum xi
    s.t. V in SOCP-relaxation
        (g,xi,eta) in PLA
    
    #input
    A: sample covariance matrix
    r: # of eigenvalues
    k: sparsity parameter
    T: maximum running time
    N: PLA coefficient

    #output
    Objective value
    '''
    d,d = A.shape

    Lambda,a = np.linalg.eig(A)    
    Lambda = Lambda.real
    a = a.real

    spca = gp.Model()
    spca.setParam('TimeLimit', 60*T)
    # Variable

    # v
    v = []
    absv = []
    for i in range(d):
        v.append(spca.addVars(r,lb = -GRB.INFINITY,ub = GRB.INFINITY, vtype = GRB.CONTINUOUS))
        absv.append(spca.addVars(r,lb = 0,ub = GRB.INFINITY, vtype = GRB.CONTINUOUS))
    # g
    g = []
    for i in range(d):
        g.append(spca.addVars(r,lb = -GRB.INFINITY,ub = GRB.INFINITY, vtype = GRB.CONTINUOUS))
    # xi
    xi = []
    for i in range(d):
        xi.append(spca.addVars(r,lb = 0,ub = GRB.INFINITY, vtype = GRB.CONTINUOUS))
    # xi
    eta = [[0 for i in range(r)] for j in range(d)]
    for j in range(d):
        for i in range(r):  
            eta[j][i] = spca.addVars(2*N+1,lb = [0]*(2*N+1),ub = [1]*(2*N+1),vtype = GRB.CONTINUOUS)

    # coefficient

    # theta
    theta = []
    for j in range(d):
        order = np.argsort(abs(np.array(a[:,j])))
        theta.append(math.sqrt(sum(abs(a[order[-1-i],j]**2)  for i in range(k))))

    # gamma
    gamma = [[[0 for h in range(2*N+1)] for i in range(r)] for j in range(d)]
    for j in range(d):
        for i in range(r):
            for h in range(2*N+1):
                gamma[j][i][h] = -theta[j]+h*(2*theta[j]/(2*N))

    # constraint

    # PLA
    for j in range(d):
        for i in range(r):
            spca.addConstr(g[j][i] == sum((a[h][j]*v[h][i]) for h in range(d)))
            spca.addConstr(g[j][i] == sum((gamma[j][i][l]*eta[j][i][l]) for l in range(2*N+1)))
            spca.addConstr(xi[j][i] == sum(((gamma[j][i][l]**2)*eta[j][i][l]) for l in range(2*N+1)))
            spca.addSOS(GRB.SOS_TYPE2, eta[j][i])

    # SOCP
    for i in range(r):
        spca.addConstr(sum(v[h][i]**2 for h in range(d))<=1)
    for i1 in range(r):
        for i2 in range(i1+1,r):
            spca.addConstr(sum((v[h][i1]+v[h][i2])**2 for h in range(d))<=2)
            spca.addConstr(sum((v[h][i1]-v[h][i2])**2 for h in range(d))<=2)

    l2norm = spca.addVars(d,lb = 0,ub = 1,vtype = GRB.CONTINUOUS)
    for j in range(d):
        spca.addConstr(l2norm[j]*l2norm[j] == sum((v[j][h]**2) for h in range(r)))
    spca.addConstr(sum((l2norm[j]) for j in range(d))<= math.sqrt(r*k))
    spca.params.NonConvex = 2

    for j in range(d):
        for i in range(r):
            spca.addConstr(absv[j][i] == gp.abs_(v[j][i]))
    for j in range(r):
        spca.addConstr(sum((absv[h][j]) for h in range(d)) <= math.sqrt(k))

    # obj
    obj = sum((Lambda[j]*sum(xi[j][i] for i in range(r))) for j in range(d))
    spca.setObjective(obj, GRB.MAXIMIZE)

    spca.optimize()
    objval =  spca.getObjective()
    return objval.getValue()

def upperbound_matrix(A,r,k,T=120,N=200):
    '''
    PLA with SOCP-relaxation
    In matrix form , sometimes faster for large instance

    max sum Lambda sum xi
    s.t. V in SOCP-relaxation
        (g,xi,eta) in PLA
    
    #input
    A: sample covariance matrix
    r: # of eigenvalues
    k: sparsity parameter
    T: maximum running time
    N: PLA coefficient

    #output
    Objective value
    '''
    d = A.shape[0]
    Lambda,a = np.linalg.eig(A)    
    Lambda = Lambda.real
    a = a.real
    Lambda = Lambda.real
    a = a.real

    spca = gp.Model()
    spca.setParam('TimeLimit', 60*T)

    # Constant

    # theta
    theta = []
    for j in range(d):
        order = np.argsort(abs(np.array(a[:,j])))
        theta.append(math.sqrt(sum(abs(a[order[-1-i],j]*a[order[-1-i],j])  for i in range(k))))
    # gamma
    gamma = [[[0 for h in range(2*N)] for i in range(r)] for j in range(d)]
    for j in range(d):
        for i in range(r):
            for h in range(2*N):
                gamma[j][i][h] = -theta[j]+h*(2*theta[j]/(2*N-1))
                
    # Variable

    # v
    v = spca.addMVar((d,r),lb = -GRB.INFINITY, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "v")
    # absv
    absv = spca.addMVar((d,r),lb = 0, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "absv")
    for i,j in zip(v.tolist(),absv.tolist()):
        spca.addConstr( j[0] == gp.abs_(i[0]))
    # g not necessary
    g = spca.addMVar((d,r),lb = -GRB.INFINITY, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "g")
    # xi
    xi = spca.addMVar((d,r),lb = 0, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "xi")
    # eta, y
    eta = [[0 for i in range(r)] for j in range(d)]
    y = [[0 for i in range(r)] for j in range(d)]
    for j in range(d):
        for i in range(r):
            eta[j][i] = spca.addMVar(2*N, lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name = "eta")
            y[j][i] = spca.addMVar(2*N-1, vtype=GRB.BINARY, name = "y")
            spca.addSOS(GRB.SOS_TYPE2, eta[j][i].tolist())

    # PLA
    for j in range(d):
        for i in range(r):
            spca.addConstr(a[:,j] @ v[:,i] == g[j][i])
            spca.addConstr(g[j][i] == sum((gamma[j][i][l]*eta[j][i][l]) for l in range(2*N)))
            spca.addConstr(xi[j][i] == sum(((gamma[j][i][l]**2)*eta[j][i][l]) for l in range(2*N)))

    # CR2
    for j in range(r):
        spca.addConstr(v[:,j] @ v[:,j] <=1)
    for i1 in range(r):
        for i2 in range(r):
            if i1 != i2:
                spca.addConstr((v[:,i1]+v[:,i2]) @ (v[:,i1]+v[:,i2]) <=2)
                spca.addConstr((v[:,i1]-v[:,i2]) @ (v[:,i1]-v[:,i2]) <=2)
    for j in range(r):
        spca.addConstr(absv[:,j].sum() <=math.sqrt(k))

    l2norm = spca.addMVar(d,lb = 0, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "l2norm")
    for i in range(d):
        spca.addConstr(l2norm[i]@l2norm[i] == v[i,:]@v[i,:])
    spca.addConstr(sum(l2norm[i] for i in range(d))<= math.sqrt(r*k))
    spca.params.NonConvex = 2


    obj = sum((Lambda[j]*sum(xi[j,i] for i in range(r))) for j in range(d))
    spca.setObjective(obj, GRB.MAXIMIZE)

    spca.optimize()
    objval =  spca.getObjective()
    return objval.getValue()