import numpy as np

def SOR(a,b,x0,w=1,c=0.0001,d=30):
    x1=x0.copy()*1.0
    x2=x0.copy()*1.0
    k=0
    
    while k<d:
        k=k+1
        for i in range(a.shape[1]):
            x2[i]=(-a[i].dot(x2)+b[i])*w/a[i,i]+x2[i]
        if np.max(np.abs(x2-x1))<=c:
            break
        x1=x2.copy()
    if(k==d):
      info='nao convergiu'
    else:
      info='convergiu'
    k=k-1
    return x2,info,k
