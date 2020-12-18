import cv2
import numpy as np
from PIL import Image
def Check(img,tag,depth=8):
    max=np.max(img)
    min=np.min(img)
    print('第',tag,'张: max=',max,' min=',min)
    if max<2**8-1 and min>10:
        return False,max,min
    return False,max,min



DEPTH=8
if __name__=='__main__':

    crop_scale=800


    img0=cv2.imread(f'./img/0.bmp',cv2.IMREAD_UNCHANGED)
    w,h,ch=img0.shape
    img0=img0[int(w/2-crop_scale):int(w/2+crop_scale),int(h/2-crop_scale):int(h/2+crop_scale),:]
    flag,max0,min0=Check(img0,0,depth=DEPTH)
    if flag:
        print('base pm_pic: OK!')
    else:
        print('base pm_pic: not OK!')
    for i in range(1,7):
        img=cv2.imread(f'./img/{i}.bmp',cv2.IMREAD_UNCHANGED)
        img=img[int(w/2-crop_scale):int(w/2+crop_scale),int(h/2-crop_scale):int(h/2+crop_scale),:]
        Flag,max,min= Check(img,i,depth=DEPTH)
        if Flag:
            print(i,'-th pic: OK!')
        else:
            print(i,'-th pic: not OK!')
        if abs(int(min0)-int(min))>100:
            print('-----',i,'-th pic: 过曝！')

