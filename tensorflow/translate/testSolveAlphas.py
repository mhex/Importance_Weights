
import numpy as np
import solveAlphas as sa

M = np.random.rand(50, 5)

K = np.dot(np.transpose(M), M)

print(K)

error = np.ones(5)

print(error)

alphas = sa.solveAlphas(K, 0.1, 0.1, error)

print(alphas)