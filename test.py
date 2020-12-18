import cv2
import numpy as np
from scipy import signal
from PIL import Image
from scipy import sparse

debug=0
dst=10
def maskBinary(mask):
    mask[mask<240]=0
    mask[mask!=0]=1
    mask=mask.astype(np.bool)
    return mask

def showImg(img):
    cv2.namedWindow('Show',0)
    cv2.resizeWindow('Show',230,320)
    cv2.imshow('Show',img)
    cv2.waitKey(0)
    return
def FilterLaplace(img,back_img,valid='same'):
    Laplace=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    print('img shape:',img.shape)
    print('backImg shape:',back_img.shape)

    h,w,ch=img.shape
    back_img[dst:h+dst,dst:w+dst,:]=img[:,:,:]
    imgB = back_img[:, :,0]
    imgG = back_img[:, :, 1]
    imgR = back_img[:, :, 2]
    print('imgR shape:',imgB.shape)
    # IMAGE=np.zeros(4416*2488*3)
    # IMAGE=IMAGE.astype(np.uint8)
    # IMAGE=np.reshape(IMAGE,[4416,2488,3])
    # IMAGE[:,:,0]=IMAGE[:,:,0]+imgB
    # IMAGE[:,:,1]=IMAGE[:,:,1]+imgG
    # IMAGE[:, :, 2] = IMAGE[:,:,2]+imgR
    # #print('img B shape',imgB.shape)
    # showImg(IMAGE)
    #print('difference:',IMAGE-img)


    LaplaceImagB=signal.convolve2d(imgB,Laplace,valid)
    LaplaceImagG = signal.convolve2d(imgG, Laplace, valid)
    LaplaceImagR = signal.convolve2d(imgR, Laplace, valid)
    print('Laplace shape:',LaplaceImagB.shape)


    # back_img[1:h-1,1:w-1,0]=LaplaceImagB
    # back_img[1:h - 1, 1:w - 1, 1] = LaplaceImagG
    # back_img[1:h - 1, 1:w - 1, 2] = LaplaceImagR

    # B= back_img[:,:,0]
    # G= back_img[:,:,1]
    # R = back_img[:, :, 2]
    print('return laplace shape B:',LaplaceImagB.shape)
    return LaplaceImagB,LaplaceImagG,LaplaceImagR


def GenerateA(h,w):
    A = np.zeros(h * w * h * w)
    A = np.reshape(A, [h * w, w * h])
    for i in range(0,h):
        for j in range(0,w):
            a = np.zeros(h * w)
            a=np.reshape(a,[h,w])
            if i==0 or j==0 or i==h-1 or j==w-1:
                a[i][j]=1
            else :
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
            #print('A[]：',A[i*w+j])
    print(A)
    print('Gerneate A shape:',A.shape)
    A_compress = sparse.csr_matrix(A)
    sparse.save_npz('I.npz', A_compress)
def SolveF(convoledG,h,w):
    A_sparse = sparse.load_npz('I.npz')
    A = A_sparse.toarray()
    Tag=357
    print('A[',Tag,'] !=0:',A[Tag][A[Tag]!=0])
    g=convoledG.flatten()
    print('g:',g)
    print('A shape:',A.shape,'g shape:',g.shape)
    X=np.linalg.solve(A,g)
    print('X',X)
    F=np.reshape(X,[h+2*dst,w+2*dst])
    print(F)
    return F
def SourceImg(img,back_img,mask):

    #img=img*mask
    # back_img = back_img[50:50 + h, 50:50 + w, :]
    # mask_bk=~mask
    # back_img=back_img*mask_bk
    #
    # img=img+back_img
    back_img = back_img[50-dst:50 + h+dst, 50-dst:50+dst + w, :]
    B,G,R=FilterLaplace(img,back_img)
    # h, w, ch = img.shape
    #
    #back_img=back_img[50:50 + h, 50:50 + w, :]
    #
    # Bb,Gb,Rb=FilterLaplace(back_img,'same')


    FB=  SolveF(B, h, w)
    FG = SolveF(G, h, w)
    FR = SolveF(R, h, w)

    IMAGE = np.zeros((h+2*dst)*(w+2*dst)*3)

    IMAGE = np.reshape(IMAGE, [h+2*dst,w+2*dst,3])
    IMAGE[:, :, 0] = FB
    IMAGE[:, :, 1] = FG
    IMAGE[:, :, 2] = FR
    #showImg(IMAGE)
    cv2.imwrite('readycover.jpg',IMAGE)
    return IMAGE
def Fusion(img):
    newImage=cv2.imread('bg.jpg')
    h,w,ch=img.shape
    print('Fusion img shape :',img.shape)
    newImage[50-dst:50+h-dst,50-dst:50+w-dst,:]=img
    #showImg(newImage)
    showImg(newImage)
    cv2.imwrite('Final.jpg', newImage)



if __name__=='__main__':
    img = cv2.imread('I.jpg')
    mask=cv2.imread('I_mask.jpg')
    back_img = cv2.imread('bg.jpg')
    mask=maskBinary(mask)
    #cv2.imwrite('sdsd.jpg',img)
    h,w,ch=img.shape
    print(img.shape)
    GenerateA(h+2*dst,w+2*dst)
    #img2=cv2.imread('readycover.jpg')
    # img=mask*img
    img=SourceImg(img,back_img,mask)

    Fusion(img)

