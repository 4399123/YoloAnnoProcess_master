from glob import glob
import os
import shutil


inputpath=r'./o_imgs'
outpath=r'./images'

if not os.path.exists(outpath):
    os.makedirs(outpath)

jsonpaths=glob(os.path.join(inputpath,'*.json'))

for jsonpath in jsonpaths:
    basename=os.path.basename(jsonpath)
    name=basename.replace('.json','.png')
    shutil.copy(jsonpath.replace('.json','.png'),os.path.join(outpath,name))