# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse
import sys

'''
    Implementação baseada na documentaçao do openCV:
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
'''

class Segment:
        # construtor da classe com default de clusters = 5.
    def __init__(self,segments=5):
        self.segments=segments

        # Método para segmentar a imagem usando K-means.
    def kmeans(self,image):
        image       =   cv2.GaussianBlur(image,(7,7),0)
        vectorized  =   image.reshape(-1,3)
        vectorized  =   np.float32(vectorized) 
        criteria    =   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(vectorized,self.segments,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))
        return label.reshape((image.shape[0],image.shape[1])),segmented_image.astype(np.uint8)

        # Extração do K da imagem segmentada.
    def extractComponent(self,image,label_image,label):
        component=np.zeros(image.shape,np.uint8)
        component[label_image==label]=image[label_image==label]
        return component


if __name__ == "__main__":
   
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Caminho da Imagem")
    ap.add_argument("-n", "--segments", required = False, type = int,help = "nº de Clusters")
    args = vars(ap.parse_args())

    image=cv2.imread(args["image"])
    if len(sys.argv)==3:        
        seg = Segment()
        label,result= seg.kmeans(image)
    else:
        seg=Segment(args["segments"])
        label,result=seg.kmeans(image)
    cv2.imshow("segmenteda",result)
    imageString = args["image"].split("/") 
        
    cv2.imwrite("Segmentada_"+imageString[0],result)
    
    for cluster in range(args["segments"]):
        result=seg.extractComponent(image, label, cluster)
        cv2.imwrite("Component_"+str(cluster)+"_"+imageString[0], result)
    
    #cv2.imshow("extraida",result)
    #cv2.waitKey(0)

#TODO -1: Rotina para segmentar as imagens contidas em uma dada pasta.
#TODO -2: Refatorar o código e otimizar o tempo.
