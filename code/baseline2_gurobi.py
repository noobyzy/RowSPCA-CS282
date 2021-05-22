import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math

def baseline2(A,k,r):
    '''
    Not supported by Gurobi now

    max Tr(AP)
    s.t. P,I-P are PSD
        Tr(P) = r
        1^T|P|1 <= rk

    #input
    A: sample covariance matrix
    r: # of eigenvalues
    k: sparsity parameter

    #output
    Objective value
    '''
    d = A.shape[0]
    bl2 = gp.Model()

    # P
    P = []
    for i in range(d):
        P.append(bl2.addVars(d,lb = -GRB.INFINITY, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "P"))

    # Px = lambda x with lambda >= 0
    x = []
    Lambda = []
    for i in range(d):
        x.append(bl2.addVars(d,lb = -GRB.INFINITY, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "x"))
        Lambda.append(bl2.addVars(1,lb = 0, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "x"))
    for i in range(d):
        for j in range(i+1,d):
            bl2.addConstr(Lambda[i] != Lambda[j])
    Ax = []
    for i in range(d):
        Ax.append(bl2.addVars(d,lb = -GRB.INFINITY, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "P"))
    for k in range(d):
        for i in range(d):
            bl2.addConstr(sum(x[k][j]*P[i][j] for j in range(d)) == Ax[k][i])
            bl2.addConstr(Lambda[k]*x[k][j] == Ax[k][i])

    # I-P
    I_P = []
    for i in range(d):
        I_P.append(bl2.addVars(d,lb = -GRB.INFINITY, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "I_P"))
    for i in range(d):
        for j in range(d):
            if i == j:
                bl2.addConstr(I_P[i][j] == 1 - P[i][j])
            else:
                bl2.addConstr(I_P[i][j] == -P[i][j])

    # (I-P)x = lambda x with lambda >= 0
    xI = []
    LambdaI = []
    for i in range(d):
        xI.append(bl2.addVars(d,lb = -GRB.INFINITY, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "x"))
        LambdaI.append(bl2.addVars(1,lb = 0, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "x"))
    for i in range(d):
        for j in range(i+1,d):
            bl2.addConstr(LambdaI[i] != LambdaI[j])
    AxI = []
    for i in range(d):
        AxI.append(bl2.addVars(d,lb = -GRB.INFINITY, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "P"))
    for k in range(d):
        for i in range(d):
            bl2.addConstr(sum(xI[k][j]*I_P[i][j] for j in range(d)) == AxI[k][i])
            bl2.addConstr(LambdaI[k]*xI[k][j] == AxI[k][i])

    # Tr(p) = r
    bl2.addConstr(sum(P[i][i] for i in range(d)) ==  r)

    # 1 |P| 1 <= rk
    absP = []
    for i in range(d):
        absP.append(bl2.addVars(d,lb = 0, ub = GRB.INFINITY,vtype=GRB.CONTINUOUS,name = "absP"))
    for j in range(d):
        for i in range(d):
            bl2.addConstr(absP[j][i] == gp.abs_(P[j][i]))
    bl2.addConstr(sum(sum (absP[j][i] for i in range(d)) for j in range(d)) <= r*k)

    # Tr(AP)
    obj = sum(sum(A[i][j]*P[j][i] for j in range(d)) for i in range(d))
    bl2.setObjective(obj, GRB.MAXIMIZE)
    bl2.params.NonConvex = 2
    bl2.optimize()
