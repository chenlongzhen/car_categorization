#encoding=utf-8

'''
Description     :This script makes car series predictions.
Author          :chenlongzhen
usage           :python load_model.py
'''

import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


'''
Image processing helper function
'''
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
#    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
#    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
#    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
#mean_blob = caffe_pb2.BlobProto()
#with open('../data/imagenet_mean.binaryproto') as f:
#    mean_blob.ParseFromString(f.read())
#mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
#    (mean_blob.channels, mean_blob.height, mean_blob.width))
#print(mean_array)

mu = np.load('../data/ilsvrc_2012_mean.npy')
mean_array = mu.mean(1).mean(1) # average over pixels to obtain the mean (BGR) pixel values  
print 'mean-subtracted values:', zip('BGR',mu) 

#Read model architecture and trained model's weights
net = caffe.Net('deploy.prototxt',
                '../data/googlenet_finetune_web_car_iter_10000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
name class mapping
'''

label_name_dic={}
with open('../data/name_class.csv') as infile:
    for num,line in enumerate(infile):
        segs = line.strip().split(',')
        label = int(segs[0])
        brand = segs[1]
        series = segs[2]
        label_name_dic[label] = "{},{}".format(brand,series)
print("{} labels read".format(num+1))
        

'''
Making predicitions
'''
##Reading image paths
test_img_paths = [img_path for img_path in glob.glob("../data/cars_train/*jpg")]
print("{} pics".format(len(test_img_paths)))

test_ids = []
preds = []


#Making predictions
for num,img_path in enumerate(test_img_paths):
    if num % 1000 == 0:
        print("{} processed".format(num))
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    preds = preds + [pred_probas.argmax()]

#    print img_path
#    print pred_probas.argmax()
#    print label_name_dic[pred_probas.argmax()]
#    print '-------'


'''
Making submission file
'''
with open("../data/output/out_mean.csv","w") as f:
    #f.write("id,label\n")
    for i in range(len(test_ids)):
        f.write(str(test_ids[i])+","+str(preds[i])+","+label_name_dic[preds[i]]+"\n")
f.close()
