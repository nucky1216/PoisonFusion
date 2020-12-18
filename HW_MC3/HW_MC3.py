import numpy as np
import random
import fractions
def GenerateMatrixAndLabal(m,n):
    N=20
    Tyr=0
    while 1:
        A=np.random.randint(1,5,(m,n))
        b=np.random.randint(1,5,(m,))

        #create linear sovle
        #solve=random.sample(range(0,N),N)
        Alphas=np.random.randint(5,10,(n,))

        b=np.zeros((m,))
        for i in  range(n):

            alpha=Alphas[i]
            b=b+alpha*A[:,i]

        Ab=np.insert(A,n,b,axis=1)
        rankA=np.linalg.matrix_rank(A)
        rankAb=np.linalg.matrix_rank(Ab)
        print('A rank:',rankA)
        print('Ab rank:',rankAb)
        print('Try Generate time:',Tyr)
        Tyr+=1
        if rankAb==rankA:
            np.savetxt('./data.txt',A)
            np.savetxt('./label.txt',b)
            np.savetxt('./alphas.txt',Alphas)
            break
    return True

def NormL2(L):
    lenth=len(L)
    sum=0
    for i in range(lenth):
        sum+=L[i]**2
    return  np.sqrt(sum)
def GradiantNormal(A,X,y):
    gradiant = 2 * np.dot(A.T, np.dot(A, X) - y)
    gn=np.linalg.norm(gradiant)
    return gn
def lossF(A,X,y):
    norm= np.linalg.norm(y-A.dot(X))
    return norm**2
def LeastQuare(A,y,step):

    X=np.zeros(N)
    # A=np.loadtxt('data.txt')
    # y=np.loadtxt('label.txt')
    w,h=A.shape
    Treshhold=0.000001
    times=50000
    flps=0
    GradientNorm=0
    while 1:
        # print('Ax',np.dot(A, X))
        # print('A *x -y',np.dot(A, X) - y)
        gradiant=2*np.dot(A.T,np.dot(A,X)-y)
        flps+=(w+w-1)*h+h+h
        #print('gradiant:',gradiant)

        update=X-step*gradiant
        flps+=h+h*w
        L2 = NormL2(X-update)
        X=update
        times=times-1
        print('Times:',times)
        #print('update:',update)

        print('L2:',L2)
        if L2<Treshhold:
            GradientNorm=np.linalg.norm(gradiant)
            break
        if times<=0:
            GradientNorm = np.linalg.norm(gradiant)
            break
    return X,flps,GradientNorm


def ModifiedGramSchmidt(A):
    Q = np.copy(A)
    w,h=A.shape
    for i in range(h):
        v=Q[:,i]
        norm = v/np.linalg.norm(v)
        Q[:, i]=norm
        for j in range(i+1,h):
            Q[:,j]=Q[:,j]-np.dot(Q[:,j].dot(norm),norm)
    return Q
def NormalF(Q):
    r,c=Q.shape
    I=np.identity(r)
    Q=Q.T.dot(Q)-I
    sum=0
    for i in range(r):
        for j in range(c):
            sum+=Q[i][j]**2
    return np.sqrt(sum)

def cholsky2(A,y):
    A2=A.copy()
    A=A.T.dot(A)
    L=np.zeros(A.shape)
    w,h=A.shape
    flps=0
    for i in range(w):
        if i==0:
            L[i,i]=np.sqrt(A[i,i])
            flps+=1
            L[i+1:,i]=A[i+1:,i]/L[i,i]
            flps += w-1
            #print('L[i+1:,i]:',L[i+1:,i])
            continue
        L[i,i]=np.sqrt(A[i,i]-L[i,:i-1].T.dot(L[i,:i-1]))
        flps+=1+1+(i-1)+i-1-1
        # print('L[i,i]:',L[i,i])
        # print(L[i,i])
        for j in range(i+1,w):
            L[j,i]=(A[j,i]-L[j,:i-1].T.dot(L[i,:i-1]))/L[i,i]
            flps +=1+i-1+i-1-1+1
    L=np.linalg.cholesky(A)
    b = np.dot(A2.T, y)
    flps+=w*w+(w-1)*w
    # Ly=b
    y1 = np.linalg.inv(L).dot(b)
    flps +=w*(w-1)/2

    # L^Tx=y
    X = np.linalg.inv(L.T).dot(y1)
    flps += w * (w - 1) / 2

    return X,flps
def LossQ(Q):
    w,h=Q.shape
    diff=Q.T.dot(Q)-np.identity(w)
    return np.linalg.norm(diff)
def ClassicGramSchmidt(A):

    w,h=A.shape
    Q=A.copy()

    for i in range(h):
        v=Q[:,i]
        for j in range(i):
            x=Q[:,j]
            scale=np.dot(x,v)
            v=v-scale*x
        Q[:,i]=v/np.linalg.norm(v)
    return Q
if __name__=='__main__':
    M=1000
    N=210
    p=1e-9

    A=np.loadtxt('data2.txt')
    print(A.shape)
    y=np.loadtxt('label2.txt')
    #GroundTruth=np.loadtxt('alphas.txt')

    x1,flps1,GN=LeastQuare(A,y,0.00001)
    # x2,flps2=cholsky2(A,y)
    #
    # np.savetxt('sol1.txt',x1.reshape([210,1]))
    # np.savetxt('sol2.txt', x2.reshape([210, 1]))
    # lossF1 =lossF(A,x1,y)
    # lossF2=lossF(A,x2,y)
    # grad1=GradiantNormal(A,x1,y)
    # grad2 = GradiantNormal(A, x2, y)
    # print(x1[-4:],flps1,lossF1)
    # print('grad1:',grad1)
    # print(x2[-4:],flps2,lossF2)
    # print('grad2:', grad2)
    #

