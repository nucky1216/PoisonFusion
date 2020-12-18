import cv2
import numpy as np
from scipy import signal
from PIL import Image
from scipy import sparse

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
    Laplace=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    GradX=np.array([[0,-1,1]])
    GradY = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])

    Grad2X = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    Grad2Y = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    print('img shape:',img.shape)
    #print('backImg shape:',back_img.shape)

    h,w,ch=img.shape
    #back_img[1:h-1,1:w-1,:]=img[1:h-1,1:w-1,:]
    imgB = img[:, :,0]
    imgG = img[:, :, 1]
    imgR = img[:, :, 2]
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

    # img_GradX=cv2.filter2D(img,-1,GradX)
    # img_GradY =cv2.filter2D(img,-1,GradY)
    #
    # img_Grad2X=cv2.filter2D(img_GradX,-1,Grad2X)
    # img_Grad2Y = cv2.filter2D(img_GradY, -1, Grad2Y)

   # img=img_Grad2X+img_Grad2Y


    LaplaceImagB=signal.convolve2d(imgB,Laplace,valid)
    LaplaceImagG = signal.convolve2d(imgG, Laplace, valid)
    LaplaceImagR = signal.convolve2d(imgR, Laplace, valid)
    print('Laplace shape:',LaplaceImagB.shape)


    # back_img[1:h-1,1:w-1,0]=LaplaceImagB
    # back_img[1:h - 1, 1:w - 1, 1] = LaplaceImagG
    # back_img[1:h - 1, 1:w - 1, 2] = LaplaceImagR
    # LaplaceImagB = img[:,:,0]
    # LaplaceImagG =img[:,:,1]
    # LaplaceImagR =img[:,:,2]
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
    F=np.reshape(X,[h,w])
    print(F)
    return F
def SourceImg(img,back_img,mask):

    #img=img*mask
    # back_img = back_img[50:50 + h, 50:50 + w, :]
    # mask_bk=~mask
    # back_img=back_img*mask_bk
    #
    # img=img+back_img
    h,w,ch=img.shape
    back_img = back_img[50:50 + h, 50:50 + w, :]
    maskSingle=mask[:,:,0]

    print('maskSing shape:',maskSingle.shape)
    Bb,Gb,Rb=FilterLaplace(back_img)
    B, G, R = FilterLaplace(img)

    B = B * ( maskSingle)
    G = G * ( maskSingle)
    R = R * ( maskSingle)


    Bb = Bb * (~ maskSingle)
    Gb = Gb * (~ maskSingle)
    Rb = Rb * (~ maskSingle)



    IMAGE = np.zeros((h) * (w) * 3)
    IMAGE = np.reshape(IMAGE, [h, w, 3])
    IMAGE[:, :, 0] = B
    IMAGE[:, :, 1] = G
    IMAGE[:, :, 2] = R
    #showImg(IMAGE)


    # h, w, ch = img.shape
    #
    #back_img=back_img[50:50 + h, 50:50 + w, :]
    #
    # Bb,Gb,Rb=FilterLaplace(back_img,'same')


    FB=  SolveF(B+Bb, h, w)
    FG = SolveF(G+Gb, h, w)
    FR = SolveF(R+Rb, h, w)

    IMAGE = np.zeros((h)*(w)*3)

    IMAGE = np.reshape(IMAGE, [h,w,3])
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
    img=cv2.imread('readycover.jpg')
    newImage[50:50+h,50:50+w,:]=img
    #showImg(newImage)
    showImg(newImage)
    cv2.imwrite('Final.jpg', newImage)

def FindBoundary(mask2D):

    h,w,ch=mask2D.shape
    Boundary=np.zeros(mask2D.shape)

    for i in range(h):
        for j in range(w):
            if j+1<w and mask2D[i][j][0]!= mask2D[i][j+1][0]:
                Boundary[i][j]=255
            if (i-1>=0 and mask2D[i-1][j][0]==0 and mask2D[i][j][0]==1) or (i+1<h and mask2D[i+1][j][0]==0 and mask2D[i][j][0]==1):
                Boundary[i][j]=255
    showImg(Boundary)
    return Boundary
def BuiltBandGraphicMask(mask_obj,mask_manual):
    mask_obj=maskBinary(mask_obj)
    mask_manual=maskBinary(mask_manual)

    # mask_obj=mask_obj.astype(np.uint8)
    # mask_obj=mask_obj*255
    # print(mask_obj[mask_obj>0])
    # showImg(mask_obj)
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

    all_num = len(g_mask[g_mask > 0])
    print('all_num:', all_num)
    visted_num = 0

    visted = np.zeros(g_mask.shape)
    #
    # i=start_pixel[0]
    # j=start_pixel[1]

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
            TAG.append([(i, j)])
            TAG_Matrix[i][j] = visted_num

            if (visted[i + 1][j] == 0) and (g_mask[i + 1][j] != 0)  :
                Queue.append((i + 1, j))
            if (visted[i - 1][j] == 0) and (g_mask[i - 1][j] != 0):
                Queue.append((i - 1, j))
            if (visted[i][j + 1] == 0) and (g_mask[i][j + 1] != 0) :
                Queue.append((i, j + 1))
            if (visted[i][j - 1] == 0) and (g_mask[i][j - 1] !=0  ) and (i-1,j) not in Cutline:
                Queue.append((i, j - 1))

    print(TAG_Matrix)

    return TAG


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
                Graph[i][j]=  abs(CaculateDifference(idx+1,idy,img_s,img_t)-K)
        if (idx-1,idy) in TAG:
            j = TAG.index((idx - 1, idy))
            if abs(j - i) < DIVIDE:
                Graph[i][j] = abs(CaculateDifference(idx - 1, idy,img_s,img_t)-K)
        if (idx,idy-1) in TAG:
            j = TAG.index((idx, idy-1))
            if abs(j - i) < DIVIDE:
                Graph[i][j] = abs(CaculateDifference(idx, idy-1,img_s,img_t)-K)
        if (idx,idy+1) in TAG:
            j = TAG.index((idx, idy+1))
            if abs(j - i) < DIVIDE:
                Graph[i][j] = abs(CaculateDifference(idx, idy+1,img_s,img_t)-K)

    print(Graph)
    return Graph

def Dijstra(Graph):

    lenth=Graph.shape[1]
    Path=[ [] for i in range(lenth)]
    Path[0].append(0)

    PathLenth=[]
    for i in range(lenth) :
        if Graph[0][i]!=0:
            PathLenth.append(Graph[0][i])
        else:
            PathLenth.append(np.inf)
    nodes=1
    visited=np.zeros(lenth)
    while nodes!=lenth:

        shortest=min(PathLenth)
        index=PathLenth.index(shortest)
        visited[index]=1

        if nodes==1:
            Path[index].append(0)
        Path[index].append(index)
        nodes += 1
        for i in range(index+1,lenth):
            if Graph[index][i]!=0:
                if visited[i]==0:

                    if PathLenth[i]==np.inf :
                        PathLenth[i]=0

                    NewLenth=Graph[index][i]+PathLenth[index]
                    if NewLenth<PathLenth[i]:
                        Path[i]=[]
                    Path[i].extend(Path[index])
                    PathLenth[i]=NewLenth
        PathLenth[index] = np.inf
    print(Path)

    return PathLenth

def  CaculateDifference1(x,y):
    return x*100+y
def CaculateDifference(x,y,img_s,img_t):

    pixelT=img_t[x][y]
    pixelS=img_s[x][y]
    L2_t=np.sqrt(pixelT[0]**2+pixelT[1]**2+pixelT[2]**2)
    L2_s =np.sqrt(pixelS[0] ** 2 + pixelS[1] ** 2 + pixelS[2] ** 2)
    return abs(L2_s-L2_t)

def CaculateInitial_K(Boundary,target_img,source_img):

    lenth_path=len(Boundary[Boundary>0])
    sum_diff=0
    h,w =Boundary.shape
    for i in range(h):
        for j in range(w):
            if Boundary[i][j]==1:
                sum=sum+CaculateDifference(target_img[i][j],source_img[i][j])
    K=sum/lenth_path

    return K

def TagCutline(G,p1,p2):

    y_dst=abs(p1[0]-p2[0])
    x_dst=abs(p1[1]-p1[1])
    CutLine=[]
    for i in range(0,y_dst+1):
        CutLine.append([p1[0]+i,p1[1]])

    print('CutLine',CutLine)
    return CutLine



if __name__=='__main__':
    #img = cv2.imread('I.jpg')
    img_s=cv2.imread('I.jpg')
    img_t=cv2.imread('bg.jpg')
    mask_obj=cv2.imread('I_obj_mask.jpg')
    mask_manual = cv2.imread('I_manual_mask.jpg')
    # back_img = cv2.imread('bg.jpg')
    # mask=maskBinary(mask)
    # #cv2.imwrite('sdsd.jpg',img)
    # h,w,ch=img.shape
    # print(img.shape)
    # GenerateA(h,w)
    # #img2=cv2.imread('readycover.jpg')
    # # img=mask*img
    # img=SourceImg(img,back_img,mask)
    #
    # Fusion(img)

    mask=maskBinary(mask_manual)
    CircleBeltMask=BuiltBandGraphicMask(mask_obj, mask_manual)

    G_mask=maskBinary(CircleBeltMask[:,:,0])
    print(G_mask[3][34])
    Cutline=TagCutline(CircleBeltMask[0],(1,34),(3,34))
    Tags=TagPixelBFS(Cutline, G_mask)
    Boundary_manual=FindBoundary(mask_manual)
    K=CaculateInitial_K(Boundary_manual, img_t, img_s)
    print("K:",K)
    Graphic=BuildGraphic(Tags,img_s,img_t,K)
    Dijstra(Graphic)


