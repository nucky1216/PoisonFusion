import cv2
import numpy as np
from scipy import signal
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import spsolve

import time
debug=0
dst=10
def maskBinary(mask):
    mask2=mask.copy()
    mask2[mask2<200]=0
    mask2[mask2!=0]=1
    mask2=mask2.astype(np.bool)
    return mask2

def showImg(img):
    cv2.namedWindow('Show',0)
    cv2.resizeWindow('Show',230,320)
    cv2.imshow('Show',img)
    cv2.waitKey(0)
    return
def FilterLaplace(img,valid='same'):
    Laplace=np.array([[0,1,0],
                      [1,-4,1],
                      [0,1,0]])
    # GradX=np.array([[0,-1,1]])
    # GradY = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    #
    # Grad2X = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    # Grad2Y = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    print('img shape:',img.shape)
    #print('backImg shape:',back_img.shape)

    h,w,ch=img.shape
    #back_img[1:h-1,1:w-1,:]=img[1:h-1,1:w-1,:]
    imgB = img[:, :,0]
    imgG = img[:, :, 1]
    imgR = img[:, :, 2]
    print('imgR shape:',imgB.shape)


    LaplaceImagB = signal.convolve2d(imgB, Laplace, valid)
    LaplaceImagG = signal.convolve2d(imgG, Laplace, valid)
    LaplaceImagR = signal.convolve2d(imgR, Laplace, valid)
    print('Laplace shape:',LaplaceImagB.shape)


    print('return laplace shape B:',LaplaceImagB.shape)
    return LaplaceImagB,LaplaceImagG,LaplaceImagR


def GenerateA(h,w):
    begin=time.time()
    A = np.zeros(h * w * h * w,dtype=np.int8)
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
            #print('A[]：',A[i*w+j])
    print('Generated A shape:',A.shape)
    print('Gerneate A shape:',A.shape)
    A_compress = sparse.csr_matrix(A)
    #sparse.save_npz('./Sparse/Bear3.npz', A_compress)
    sparse.save_npz('I.npz', A_compress)
    print('================Time cost:',time.time()-begin,'===========')
def SolveF(convoledG,h,w):
    A_sparse = sparse.load_npz('I.npz')
    #A_sparse = sparse.load_npz('./Sparse/Bear3.npz')
    A = A_sparse
    print('A shape:',A.shape,)
    g=convoledG.flatten().astype(np.int16)

    print('A shape:',A.shape,'g shape:',g.shape,'A dytpe',A.dtype,'g dtpe',g.dtype)
    #X=np.linalg.solve(A,g).astype(np.uint)
    X=spsolve(A,g)
    F=np.reshape(X,[h,w])
    return F
def SourceImg(img,back_img,mask,pos):

    h,w,ch=img.shape

    px,py=pos
    back_img = back_img[py:py + h, px:px + w, :]
    maskSingle=mask[:,:,0]

    print('maskSingle shape:',maskSingle.shape)
    #showImg(back_img )
    Bb,Gb,Rb=FilterLaplace(back_img)
    B, G, R = FilterLaplace(img)

    B = B * ( maskSingle)
    G = G * ( maskSingle)
    R = R * ( maskSingle)

    print('Bb shape:', Bb.shape)
    Bb = Bb * (~ maskSingle)
    Gb = Gb * (~ maskSingle)
    Rb = Rb * (~ maskSingle)

    # IMAGE = np.zeros((h) * (w) * 3)
    # IMAGE = np.reshape(IMAGE, [h, w, 3])
    #
    # IMAGE[:, :, 0] = B+Bb
    # IMAGE[:, :, 1] = G+Gb
    # IMAGE[:, :, 2] = R+Rb
    # showImg(IMAGE)

    IMAGE2 = np.zeros((h) * (w) * 3)
    IMAGE2 = np.reshape(IMAGE2, [h, w, 3])
    IMAGE2[:, :, 0] = Bb
    IMAGE2[:, :, 1] = Gb
    IMAGE2[:, :, 2] = Rb
    #showImg(IMAGE2)

    h, w, ch = img.shape

    FB=  SolveF(Bb+B, h, w)
    FG = SolveF(Gb+G, h, w)
    FR = SolveF(Rb+R, h, w)

    IMAGE = np.zeros((h)*(w)*3)

    IMAGE = np.reshape(IMAGE, [h,w,3])
    IMAGE[:, :, 0] = FB
    IMAGE[:, :, 1] = FG
    IMAGE[:, :, 2] = FR
    #showImg(IMAGE)
    cv2.imwrite('readycover.jpg',IMAGE)
    print('=========================Equation has Solved======================')
    return IMAGE
def Fusion(img,img_bg,pos):
    px,py=pos
    newImage =img_bg# cv2.imread('./Input/beach.png')
    h,w,ch=img.shape
    print('Fusion img shape :',img.shape)
    img=cv2.imread('readycover.jpg')
    newImage[py:py+h,px:px+w,:]=img
    cv2.imwrite('./Output/Final.jpg', newImage)

def FindBoundary(mask2D):

    h,w,ch=mask2D.shape
    Boundary=np.zeros(mask2D.shape)

    for i in range(h):
        for j in range(w):
            if j+1<w and mask2D[i][j][0]!= mask2D[i][j+1][0]:
                Boundary[i][j]=255
            if (i-1>=0 and mask2D[i-1][j][0]==0 and mask2D[i][j][0]==1) or (i+1<h and mask2D[i+1][j][0]==0 and mask2D[i][j][0]==1):
                Boundary[i][j]=255
    #showImg(Boundary)
    return Boundary
def BuiltBandGraphicMask(mask_obj,mask_manual):
    mask_obj=maskBinary(mask_obj)
    mask_manual=maskBinary(mask_manual)


    print(mask_manual[mask_manual==1])
    mask_manual_inv=mask_manual
    mask_manual_inv=mask_manual_inv.astype(np.uint8)
    mask_manual_inv=mask_manual_inv*255
    print(mask_obj[mask_obj>0])
    #showImg(mask_manual_inv)

    mask=~(mask_obj+~mask_manual)
    mask=mask.astype(np.uint8)
    mask=mask*255

    #showImg(mask)
    cv2.imwrite('BeltMask.jpg',mask)
    return mask

def TagPixelBFS(Cutline, g_mask):

    lenthCutLine=len(Cutline)
    start=0
    start_pixel =(Cutline[start][0],Cutline[start][1]+1)
    print('start_pixel:',start_pixel)
    all_num = len(g_mask[g_mask > 0])
    print('all_num:', all_num)
    visted_num = 0

    visted = np.zeros(g_mask.shape)


    TAG_Matrix = np.zeros(g_mask.shape)
    TAG = []
    Queue = []
    Queue.append((start_pixel[0], start_pixel[1]))
    cur = 0
    while cur <len(Queue):

        out_queue = Queue[cur]
        cur += 1
        i = out_queue[0]
        j = out_queue[1]
        if visted[i][j] == 0:
            visted_num += 1
            visted[i][j] = 1
            TAG.append((i, j))
            TAG_Matrix[i][j] = visted_num

            if (visted[i + 1][j] == 0) and (g_mask[i + 1][j] != 0)  :
                Queue.append((i + 1, j))
            if (visted[i - 1][j] == 0) and (g_mask[i - 1][j] != 0):
                Queue.append((i - 1, j))
            if (visted[i][j + 1] == 0) and (g_mask[i][j + 1] != 0) :
                Queue.append((i, j + 1))
            if (visted[i][j - 1] == 0) and (g_mask[i][j - 1] !=0  ) and (i,j-1) not in Cutline:
                Queue.append((i, j - 1))
    print('TAG_Matrix shape:', TAG_Matrix.shape)
    print('TAG_Matrix:',TAG_Matrix)
    np.savetxt('TAG_Matrix.txt',TAG_Matrix,fmt='%5d')
    return TAG,TAG_Matrix


def BuildGraphic(TAG,img_s,img_t,K):

    lenth=len(TAG)
    DIVIDE=lenth/2
    Graph=np.zeros([lenth,lenth])
    print('Graph shape:',Graph.shape)

    for i in range(lenth):

        idx,idy=TAG[i]

        if (idx+1,idy) in TAG:
            j=TAG.index((idx+1,idy))
            if abs(j - i) < DIVIDE:
                Graph[i][j]=  abs(CaculateDifference(img_s[idx+1][idy],img_t[idx+1][idy])-K)
        if (idx-1,idy) in TAG:
            j = TAG.index((idx - 1, idy))
            if abs(j - i) < DIVIDE:
                Graph[i][j] = abs(CaculateDifference(img_s[idx - 1][idy],img_t[idx - 1][idy])-K)
        if (idx,idy-1) in TAG:
            j = TAG.index((idx, idy-1))
            if abs(j - i) < DIVIDE:
                Graph[i][j] = abs(CaculateDifference(img_s[idx][idy-1],img_t[idx][idy-1])-K)
        if (idx,idy+1) in TAG:
            j = TAG.index((idx, idy+1))
            if abs(j - i) < DIVIDE:
                Graph[i][j] = abs(CaculateDifference(img_s[idx][idy+1],img_t[idx][idy+1])-K)

    #print(Graph)
    return Graph

def Dijstra(Graph,begin=0):

    lenth=Graph.shape[1]
    Path=[ [] for i in range(lenth)]


    PathLenth=[]
    for i in range(lenth) :
        PathLenth.append(np.inf)

    PathLenth[0]=0
    nodes=0
    visited=np.zeros(lenth)
    while nodes!=lenth:

        shortest=min(PathLenth)
        index=PathLenth.index(shortest)
        visited[index]=1

        Path[index].append(index+begin)
        nodes += 1
        for i in range(lenth):
            if Graph[index][i]!=0:
                if visited[i]==0:


                    NewLenth=Graph[index][i]+PathLenth[index]

                    if NewLenth<PathLenth[i]:
                        Path[i]=[]
                        PathLenth[i]=NewLenth
                       # print('find New path! Path[i]:',Path[i])
                        Path[i].extend(Path[index])

        PathLenth[index] = np.inf
    print('Path[lenth-1]:',Path[lenth-1])

    return Path[lenth-1]


def CaculateDifference(p_img_s,p_img_t):

    pixelT=p_img_t
    pixelS=p_img_s
    L2_t=np.sqrt(pixelT[0]**2+pixelT[1]**2+pixelT[2]**2)
    L2_s =np.sqrt(pixelS[0] ** 2 + pixelS[1] ** 2 + pixelS[2] ** 2)
    return abs(L2_s-L2_t)

def CaculateInitial_K(Boundary3D,source_img,target_img):

    Boundary=Boundary3D[:,:,0]
    lenth_path=len(Boundary[Boundary>0])
    sum=0
    h,w =Boundary.shape
    for i in range(h):
        for j in range(w):
            if Boundary[i][j]==255:
                sum=sum+CaculateDifference(target_img[i][j],source_img[i][j])
    K=sum/lenth_path

    return K

def TagCutline(G,p1,p2):

    y_dst=abs(p1[0]-p2[0])
    x_dst=abs(p1[1]-p1[1])
    CutLine=[]
    for i in range(0,y_dst+1):
        CutLine.append((p1[0]+i,p1[1]))

    print('CutLine',CutLine)
    return CutLine


def DrawPath(G_mask,TagMatrix,Path):

    for i in Path:
        [[x,y]]=np.argwhere(TagMatrix==i)
        G_mask[x][y]=180
    showImg(G_mask)
    cv2.imwrite('path.jpg',G_mask)
    return G_mask

def CaculateNewPathK(Path,TagMatrix,img_s,img_t):

    sum=0
    for i in Path:
        [[x,y]]=np.argwhere(TagMatrix==i)
        sum+=CaculateDifference(img_t[x][y],img_s[x][y])
    K=sum/len(Path)
    return K


def FindBestBoundary(K,Graphic,Tags,TAG_Matrix,img_s,img_t):

    TIMES=10
    THRESHOLD=1
    while 1:
        TIMES-=1
        Graphic = BuildGraphic(Tags, img_s, img_t, K)
        Path = Dijstra(Graphic, begin=1)
        K2=CaculateNewPathK(Path,TAG_Matrix,img_s,img_t)
        print('Find best Bundary:',TIMES)
        if abs(K2-K)<THRESHOLD:
            break
        if TIMES<=0:
            break

    return Path
def ReBuiltNewMask(Path,TAG_Matrix,mask):

    Boundary=[]
    for i in Path:
        [[x,y]]=np.argwhere(TAG_Matrix==i)
        Boundary.append((x,y))
    h,w,ch=mask.shape

    NewMask=np.ones(mask.shape)
    NewMask=NewMask*255

    for i in range(h):
        for j in range(w):
            if (i,j) in Boundary:
                break
            NewMask[i][j]=0

    for i in range(h):
        for j in range(w-1,-1,-1):
            if (i,j) in Boundary:
                break
            NewMask[i][j]=0
    for i in range(h-1,-1,-1):
        for j in range(w):
            if (i,j) in Boundary:
                break
            NewMask[i][j]=0

    for i in range(h-1,-1,-1):
        for j in range(w-1,-1,-1):
            if (i,j) in Boundary:
                break
            NewMask[i][j]=0


    showImg(NewMask)
    cv2.imwrite('NewMask_i.jpg',NewMask)
    return NewMask
if __name__=='__main__':

    # #============OpenCV Poisson Merge==================
    #
    # #mask= 255*np.ones(img_s.shape,img_s.dtype)
    # begin_time=time.time()
    # img_s=cv2.imread('./Input/Bear3.jpg')
    # img_t=cv2.imread('./Input/beach.png')
    # mask=cv2.imread('./Input/Bear3_mask.jpg')
    #
    # w,h,ch=img_t.shape
    # pos=(h//2-300,w//2+200)
    # mix_clone=cv2.seamlessClone(img_s,img_t,mask,pos,cv2.MIXED_CLONE)
    # normal_clone = cv2.seamlessClone(img_s, img_t, mask, pos, cv2.NORMAL_CLONE)
    #
    #
    # cv2.imwrite('./Output/MergeByCV2_Mixed2.jpg',mix_clone)
    # cv2.imwrite('./Output/MergeByCV2_Normal2.jpg', mix_clone)
    # print('============OpenCV Time Cost:', time.time() - begin_time, '====================')
    # showImg(mix_clone)
    #
    #
    # #============Poisson Merge==================
    PoissonTime=time.time()
    # img_s=cv2.imread('I.jpg')
    # img_t=cv2.imread('bg.jpg')
    # #mask_obj=cv2.imread('I_obj_mask.jpg')
    # mask_obj = cv2.imread('I_manual_mask2.jpg')


    img_s=cv2.imread('./Input/Bear3.jpg')
    img_t=cv2.imread('./Input/beach.png')
    mask_obj=cv2.imread('./Input/Bear3_mask.jpg')


    mask=maskBinary(mask_obj)
    print(' mask shape:' , mask.shape)
    print('img_s shape:', img_s.shape)

    # h,w,_=img_s.shape
    # GenerateA(h,w)# Generate the coefficient matrix

    x=50
    y=430
    pos=(x,y)

    Readyimg = SourceImg(img_s,img_t, mask,pos)
    Fusion(Readyimg ,img_t,pos)
    print('Merge Done!')
    print('=================Time cost:',-(PoissonTime-time.time()),'s =============')
    img=cv2.imread('Output/Final.jpg')
    showImg(img)
    # #============Rebult a New Mask with Dijstra==================

    # img_s=cv2.imread('I.jpg')
    # img_t=cv2.imread('bg.jpg')
    # mask_obj=cv2.imread('I_obj_mask.jpg')
    # mask_manual = cv2.imread('I_manual_mask2.jpg')
    #
    #
    #
    # CircleBeltMask=BuiltBandGraphicMask(mask_obj, mask_manual)
    #
    # G_mask=maskBinary(CircleBeltMask[:,:,0])
    # np.savetxt('G_mask.txt',G_mask,fmt='%d')
    #
    # Cutline=TagCutline(CircleBeltMask[0],(1,34),(3,34))
    # Tags,TAG_Matrix=TagPixelBFS(Cutline, G_mask)
    #
    # mask = maskBinary(mask_manual)
    # Boundary_manual=FindBoundary(mask)
    #
    # K=CaculateInitial_K(Boundary_manual, img_t, img_s)
    # print("K:",K)
    # Graphic=BuildGraphic(Tags,img_s,img_t,K)
    #
    # Path=Dijstra(Graphic,begin=1)
    # DrawPath(CircleBeltMask, TAG_Matrix, Path)
    #
    # Path=FindBestBoundary(K, Graphic, Tags,TAG_Matrix, img_s, img_t)
    # ReBuiltNewMask(Path, TAG_Matrix, mask_manual)
    # DrawPath(CircleBeltMask, TAG_Matrix, Path)
