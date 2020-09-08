import numpy as np

############################################################
## Implementation of the Gauss Seidel algorithm
## A matrix of the linear system
## f right hand side
## x0 initial guess of the solution
############################################################

def gauss_seidel(A,f,x0,ITER_MAX = 100, tol = 1E-4,_debug_=1):
  # size of the system
  n = A.shape[0]
  # initialize the residual
  res = np.linalg.norm(f-np.dot(A,x0))
  
  # init the new vector
  x_new = np.zeros(n)

  # copy the guess
  x = np.array(x0,copy=True)

  # init niter
  niter = 0

  # loop over the
  while (res>tol) and (niter<ITER_MAX):
    # loop over all the lines
    for i in range(n):
      # initialize the sums
      sum1, sum2 = 0.0, 0.0
      # loop over the line elements
      for j in range(n):
        # if j<i we use the new values
        if j<i:
          sum1 += A[i,j]*x_new[j]
          # else we use the old ones
        elif j>i:
            sum2 += A[i,j]*x[j]
      # we store the new values
      x_new[i] = (f[i]-sum1-sum2)/A[i,i]

    # change the old solution to the new one
    x = x_new

    # compute the new residual
    res = np.linalg.norm(f-np.dot(A,x))

    # increment niter
    niter += 1

  # print the final status of the algorithm
  if niter == ITER_MAX:
    info = 0
  else:
    info = 1

  return x,info, niter  
