#encoding=utf-8
#	从mat读取文件名和class

import scipy.io as sio  
import numpy as np  
  
#matlab文件名  
fix = '/data/chenlongzhen/car_class_caffe/data/cars_train'
matfn=u'../data/devkit/cars_train_annos.mat'  
data=sio.loadmat(matfn)
data = data['annotations'][0]

for tup in data:
  #print(tup)
  clas=tup[4][0][0]
  fname=tup[5][0]
  print("{}/{} {}".format(fix,fname,clas))
