from mosek.fusion import *
import numpy as np

def abs(M, t, x):
    M.constraint(Expr.add(t,x), Domain.greaterThan(0.0))
    M.constraint(Expr.sub(t,x), Domain.greaterThan(0.0))
def norm1(M, t, x):
    u = M.variable(x.getShape(), Domain.unbounded())
    abs(M, u, x)
    M.constraint(Expr.sub(t, Expr.sum(u)), Domain.greaterThan(0.0))
def baseline2(A,k,r):
    '''
    Mosek version

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
    bl2 = Model()
    P = bl2.variable(Domain.inPSDCone(d))
    absP = bl2.variable(Domain.inPSDCone(d))
    I_P = bl2.variable(Domain.inPSDCone(d))
    Id = Matrix.eye(d)
    bl2.constraint(Expr.sub(Expr.sub(Id,P),I_P), Domain.equalsTo(0))
    bl2.constraint(Expr.sum(P.diag()),Domain.equalsTo(r))
    norm1(bl2,r*k,P.reshape(d*d))
    P.reshape(d,d)
    objExpr = Expr.sum(Expr.mulDiag(A,P))
    bl2.objective(ObjectiveSense.Maximize, objExpr)
    bl2.solve()
    return bl2.primalObjValue()


