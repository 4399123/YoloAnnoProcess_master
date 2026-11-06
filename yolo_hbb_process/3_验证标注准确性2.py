import os
from imutils import paths
import cv2
from glob import glob
import numpy as np
from tqdm import  tqdm

lbl_path=r'./labels'

out_path=r'./suoying_output'

if not os.path.exists(out_path):
    os.makedirs(out_path)

txt_paths=glob(os.path.join(lbl_path,'*.txt'))

palette = np.random.randint(0, 256, (256, 3), dtype=np.int32)
palette[0]=[0,255,0]
palette[1]=[0,0,255]
palette[2]=[255,0,0]
palette[3]=[255,255,0]
palette[4]=[255,0,255]
palette[5]=[171,130,255]
palette[6]=[155,211,255]
palette[7]=[0,255,255]
palette[8]=[107,25,88]
palette[9]=[11,60,132]

def get_corresponds(center_x, center_y, bbox_width, bbox_height,img_H,img_W):
    x1=center_x-bbox_width/2
    y1=center_y-bbox_height/2
    x2=x1+bbox_width
    y2=y1+bbox_height
    x1=max(0,x1)
    y1=max(0,y1)
    x2=min(img_W,x2)
    y2=min(img_H,y2)
    return int(x1),int(y1),int(x2),int(y2)

for txt_path in tqdm(txt_paths):
    basename=os.path.basename(txt_path)
    name=basename.split('.')[0]
    img_path=txt_path.replace('labels','images').replace('.txt','.png')
    img=cv2.imread(img_path)
    H,W=img.shape[0],img.shape[1]
    with open(txt_path,'r') as f:
        lines=f.readlines()
        nn=0
        for line in lines:
            line=line.strip()
            idxywh_list=line.split(" ")
            id=int(idxywh_list[0])
            center_x=np.float64(idxywh_list[1])*np.float64(W)
            center_y=np.float64(idxywh_list[2])*np.float64(H)
            bbox_width=np.float64(idxywh_list[3])*np.float64(W)
            bbox_height=np.float64(idxywh_list[4])*np.float64(H)
            x1,y1,x2,y2=get_corresponds(center_x, center_y, bbox_width, bbox_height,H,W)
            tt=palette[id][0]
            cv2.rectangle(img=img,rec=(x1,y1,x2-x1,y2-y1),color=(int(palette[id][0]),int(palette[id][1]),int(palette[id][2])),thickness=1,lineType=1)
            cv2.putText(img=img,text=str(nn),org=(x1,y1),color=(int(palette[id][0]),int(palette[id][1]),int(palette[id][2])),thickness=1,lineType=1,fontFace=1,fontScale=1)
            nn+=1
    cv2.imwrite(os.path.join(out_path,name+'.jpg'),img)