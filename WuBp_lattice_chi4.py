import numpy as np
import pickle

chi2 = open('chi_BB.dat', "rb")
#file.close(chi2)
chi2 = np.loadtxt(chi2, dtype = float, delimiter=' ')
print chi2

chi4 = open('chi_BBBB.dat', "rb")
chi4 = np.loadtxt(chi4, dtype = float, delimiter=' ')
print chi4

fname = 'chi2_wubp.p'
pickle.dump(chi2, open(fname, "wb"))
file.close(open(fname))

fname = 'chi4_wubp.p'
pickle.dump(chi4, open(fname, "wb"))
file.close(open(fname))
