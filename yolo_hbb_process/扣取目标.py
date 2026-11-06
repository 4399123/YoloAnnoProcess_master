import  cv2
import os
from imutils import paths
from tqdm import tqdm

path='./out/images'
outpath='./cut'

if not os.path.exists(outpath):
    os.makedirs(outpath)

imgpaths=list(paths.list_images(path))

for imgpath in tqdm(imgpaths):
    basename=os.path.basename(imgpath)
    dirname=os.path.dirname(imgpath)

    img=cv2.imread(imgpath)

    labelpath=os.path.join(dirname.replace('images', 'labels'), basename[:-4]+'.txt')
    n=0
    with open(labelpath, 'r') as f:
        lines = f.readlines()
        for line in lines:

            word_list = line.split(' ')
            center_x = float(word_list[1])
            center_y = float(word_list[2])
            width = float(word_list[3])
            height = float(word_list[4])
            xmin = int((center_x - width / 2)*img.shape[1])
            ymin = int((center_y - height / 2)*img.shape[0])
            xmax = int((center_x + width / 2)*img.shape[1])
            ymax = int((center_y + height / 2)*img.shape[0])

            cut_img = img[ymin:ymax, xmin:xmax]
            cv2.imwrite("./cut/"+basename[:-4]+"_{}.png".format(n), cut_img)
            n+=1
