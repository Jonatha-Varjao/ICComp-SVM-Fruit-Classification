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
class Kmeans:
    """
        Classe representando segmentação usando clusterição
        por k-means
    """

    def __init__(self, segments = 10):
        """
            Utilizando K=4 como default
        """
        self.segments=segments

    def segmentation(self,image):
        """
            Segmentacao utilizando clustericao com kmeans
            do openCV
        """
        vectorized       =   image.reshape(-1,3)
        vectorized       =   np.float32(vectorized) 
        criteria         =   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center = cv2.kmeans(vectorized, self.segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        res              = center[label.flatten()]
        segmented_image  = res.reshape((image.shape))
        return label.reshape((image.shape[0], image.shape[1])), segmented_image.astype(np.uint8)

    def extract_component(self,image,label_image,label):
        """
            Extração dos K da imagem
        """
        component                     = np.zeros(image.shape, np.uint8)
        component[label_image==label] = image[label_image==label]
        return component
          
    def segment_folder(self, folderName):
        """
            Segmentação da pasta contendo as imagens pre processadas 
        """
        for filename in os.listdir(folderName+"Extracted_Images/"):
            if filename.lower().endswith('.JPG') or filename.lower().endswith('.PNG') :
                print(filename)
                image = cv2.imread(folderName+"Extracted_Images/"+filename)  
                label,result= self.segmentation(image)
                #cv2.imshow("segmenteda",result)
                cv2.imwrite(folderName + "Segment_Images/kmeans/Kmeans_" + filename, result)
                # for cluster in range(seg.segments):
                #     result=seg.extractComponent(image, label, cluster)
                #     cv2.imwrite("Component_"+str(cluster)+"_"+imageString[0], result)
        #return result


#TODO   1: Refatorar o código e otimizar o tempo.
#TODO   2: Adicionar listar de folders no método de segmentação
#TODO   3: Problema no filtro Gaussiano do OpenCV (widht x height) da imagem n bate no kernel escolhido
            #  - try - catch, ou interpolar a imagem pra bater o kernel....
#TODO   4: Ajeitar o problema do filename incluir o folder e interromper a segmentação
            # '.jpg' in String ( apesar que a solução tá bem clean)

def main():
    label = []
    seg4 = Kmeans(4)
    seg5 = Kmeans(5)
    seg6 = Kmeans(6)
    label, result4 = seg4.segmentation(cv2.imread(sys.argv[1]))
    label, result5 = seg5.segmentation(cv2.imread(sys.argv[1]))
    label, result6 = seg6.segmentation(cv2.imread(sys.argv[1]))
    
    cv2.imwrite("k4.jpg", result4)
    cv2.imwrite("k5.jpg", result5)
    cv2.imwrite("k6.jpg", result6)

if __name__ == "__main__":
    main()