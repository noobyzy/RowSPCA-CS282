import numpy as np
import baseline1 as bl1
import baseline2_mosek as bl2
import localsearch as ls
import PLA_SOCP as ps
import matplotlib.pyplot as plt

A = np.loadtxt('data.txt', delimiter=',')

klist = [5,10,20,40,79]
r = 2

ub = np.array([ls.LocalSearch(A, k, r, 500, A.shape[0]) for k in klist])
lb = np.array([ps.upperbound_vector(A,r,k) for k in klist])
baseline1 = np.array(bl1.baseline1(A,klist))
baseline2 = np.array([bl2.baseline2(A,k,r) for k in klist])

ln_bl1_gap = np.log( (baseline1 - lb)/baseline1)
ln_bl2_gap = np.log( (baseline1 - lb)/baseline1)
ln_ub_gap = np.log( (baseline1 - lb)/baseline1)

plt.plot(klist,ln_bl1_gap)
plt.plot(klist,ln_bl2_gap)
plt.plot(klist,ln_ub_gap)

plt.show()
