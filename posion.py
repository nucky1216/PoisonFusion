import cv2
import numpy as np
from scipy import signal
from PIL import Image
from scipy import sparse

def showImg(img):
    cv2.namedWindow('Show',0)
    cv2.resizeWindow('Show',230,320)
    cv2.imshow('Show',img)
    cv2.waitKey(0)
    return
def maskBinary(mask):
    mask[mask!=255]=0
    mask[mask != 0] = 1
    return mask
def test(img,mask):
    mask=maskBinary(mask)
    img=img*mask

    showImg(img)


def FilterLaplace(img,valid='same'):
    Laplace=np.array([[0,1,0],[1,-4,1],[0,1,0]])

    h,w,ch=img.shape
    imgB = img[ :, :,0]
    imgG = img[:, :, 1]
    imgR = img[:, :, 2]

    LaplaceImagB=signal.convolve2d(imgB,Laplace,valid)
    LaplaceImagG = signal.convolve2d(imgG, Laplace, valid)
    LaplaceImagR = signal.convolve2d(imgR, Laplace, valid)

    #LaplaceImagB[0,:]=img[0,:,0]
    #print(LaplaceImagB[0,:,:].shape)
    print(LaplaceImagB.shape)
    return LaplaceImagB,LaplaceImagG,LaplaceImagR
def FusionBF(back,fore):
    Bb,Gb,Rb=FilterLaplace(back)
    Bf,Gf,Rf=FilterLaplace(fore)

    B=Bf
    G=Gf
    R=Rf

    h,w=B.shape
    image=np.zeros(h*w*3)
    image=np.reshape(image,[h,w,3])
    image[:,:,0]=B
    image[:, :, 1] = G
    image[:, :, 2] = R


    return B,G,R
def GenerateMat_A(h,w):
    A = np.zeros(h * w * h * w)
    A = np.reshape(A, [h * w, w * h])

    for i in range(0,h):
        for j in range(0,w):
            a = np.zeros(h * w)
            a=np.reshape(a,[h,w])
            # if i==0 or j==0 or i==h-1 or j==w-1:
            #     a[i][j]=1
            # else :
            a[i][j]=-4
            if i-1>=0:
                a[i-1][j]=1
            if j-1>=0:
                a[i][j-1] = 1
            if i+1<h:
                a[i+1][j]=1
            if j+1<w:
                a[i][j+1]=1
            a=a.flatten()
            #print('a:',a)
            A[i*w+j]=a
    print(A)
    return A
    # A_compress = sparse.csr_matrix(A)
    # sparse.save_npz('I.npz', A_compress)
def solve(A,b):
    print('A shape:',A.shape)
    h,w=b.shape
    b=b.flatten()
    print('b shape:',b.shape)
    print('b>0', b[b > 0])
    print('A[0][0]:',A[450][A[450]!=0])

    x=np.linalg.solve(A,b)
    print('x>0',x[x!=0])
    print(' h',h,'w', w)
    P=np.reshape(x,[h,w])
    return P
def NewPixel(back,fore):
    h, w, ch = back.shape
    hf,wf,chf=fore.shape
    backROI=back[0:hf,0:wf,:]
    B,G,R=FusionBF(backROI,fore)

    A = sparse.load_npz('I.npz')
    A = A.toarray()
    h,w=A.shape

    print(A)
    NewB=solve(A,B)
    print('B is oK.NewB>0',NewB[NewB>0])
    NewG=solve(A,G)
    print('G is oK')
    NewR=solve(A,R)
    print('R is oK')

    NewImage=np.zeros(h*w*ch)
    NewImage=np.reshape(NewImage,[w,h,ch])
    NewImage[0:hf,0:wf, 0]=NewB
    NewImage[0:hf,0:wf, 1] = NewG
    NewImage[0:hf,0:wf, 2] = NewR
    showImg(NewImage)
    cv2.imwrite('I_ready.jpg',NewImage)

if __name__ == '__main__':
    fore_img=cv2.imread('I.jpg')
    back_img = cv2.imread('bg.jpg')
    mask=cv2.imread('I_mask.jpg')
    h,w ,ch=fore_img.shape

    # A=GenerateMat_A(3,3)
    # b=np.array([1,2,3,4,5,6,7,8,9])
    # print(np.linalg.solve(A,b))
    a=np.array([[1,2,2],[2,34,4],[2,5,6]])
    mask=np.array([[0,0],[0,0]])
    a[1:3,1:3]=mask
    print(a)
    #test(fore_img,mask)
    print('ok')
