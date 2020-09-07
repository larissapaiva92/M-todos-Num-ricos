import numpy as np
import numpy.linalg as nl

def jacobi(A,f,x,maxIter = 100, tol = 1.0e-4):
    # inputs:
    # A is a nxn matrix
    # f is a right-hand-side vector of length n
    # x is initial guess at the solution to A x = f
    # maxIter (optional) is maximum iterations
    # tol (optional) is desired accuracy in terms
    # of the L2-norm of the residual (= f - Ax)
    n = f.size
    # Begin by checking for compatible sizes
    if (A.shape[0] != n or A.shape[1] != n):
        print("Error! Incompatible sizes.")
        return f
    # Loop to iterate until we converge to solution
    # or we reach the maximum number of iterations
    xnew = np.copy(x)
    for iter in range(maxIter):
        # calculate residual
        res = f - np.dot(A,x)
        # check L2-norm for convergence
        if (nl.norm(res,2) < tol):
            #print("Converged after", iter,"iterations ")
            return x, iter
        # start of Jacobi iteration
        for i in range(n):
            sum=0.0
            for j in range(n):
                if(i != j):
                    sum += A[i,j]*x[j]
            xnew[i] = (f[i] - sum)/A[i,i]
        x = np.copy(xnew)
        #print('Failed to converge after', iter,'iterations')
    return x,iter
