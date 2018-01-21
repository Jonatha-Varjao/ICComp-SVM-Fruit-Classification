# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse
import sys
import os

'''
    Implementação baseada na documentaçao do openCV:
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
'''

class Segment:
        # construtor da classe com default de clusters = 4.
    def __init__(self, segments = 4):
        self.segments=segments

        # Método para segmentar a imagem usando K-means.
    def kmeans(self,image):
        vectorized       =   image.reshape(-1,3)
        vectorized       =   np.float32(vectorized) 
        criteria         =   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center = cv2.kmeans(vectorized, self.segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        res              = center[label.flatten()]
        segmented_image  = res.reshape((image.shape))
        return label.reshape((image.shape[0], image.shape[1])), segmented_image.astype(np.uint8)

        # Extração do K da imagem segmentada.
    def extract_component(self,image,label_image,label):
        component                     = np.zeros(image.shape, np.uint8)
        component[label_image==label] = image[label_image==label]
        return component
        
        # Segmentação da pasta contendo as imagens pre processadas
    def segment_folder(self, folderName):
        for filename in os.listdir(folderName+"Extracted_Images/"):
            if '.' in filename:
                print(filename)
                image = cv2.imread(folderName+filename)  
                label,result= seg.kmeans(image)
                #cv2.imshow("segmenteda",result)
                cv2.imwrite(folderName + "Segment_Images/kmeans/Kmeans_" + filename, result)
                # for cluster in range(seg.segments):
                #     result=seg.extractComponent(image, label, cluster)
                #     cv2.imwrite("Component_"+str(cluster)+"_"+imageString[0], result)
        return result


#TODO   1: Refatorar o código e otimizar o tempo.
#TODO   2: Adicionar listar de folders no método de segmentação
#TODO   3: Problema no filtro Gaussiano do OpenCV (widht x height) da imagem n bate no kernel escolhido
            #  - try - catch, ou interpolar a imagem pra bater o kernel....
#TODO   4: Ajeitar o problema do filename incluir o folder e interromper a segmentação
            # '.jpg' in String ( apesar que a solução tá bem clean)
