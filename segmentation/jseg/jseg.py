# -*- coding: utf-8 -*-
import cv2
import os
import sys
from subprocess import call
from PIL import Image, ImageDraw
#from segmentation.rotulacao.rotulacao import Rotulacao


"""
 Script para utilizar o binário da segmentação do JSEG.
    Utilizando wine 3.0.4 ( para executar binário para windows )
    Scrip ira chamar o programa na qual faz a segmentação utilizando Jseg
    será segmentadas as imagens que já estão previamente pré-processadas
"""
class Jseg():
    """
        Classe representação a segmentação de uma imagem utilizando
        o algoritmo JSEG deng e manjuntah 2001
    """

    def segment_folder(self, folderName):
        """
            Segmentação da pasta contendo as imagens pre processadas
        """
        for filename in os.listdir(folderName+"Extracted_Images/"):
            if filename.lower().endswith('.JPG') or filename.lower().endswith('.PNG') :
                image = cv2.imread(folderName+"Extracted_Images/"+filename)
                print(filename, str(image.shape[0]),image.shape[1])
                call(["wine","segmentation/jseg/segwin.exe","-i",folderName+"Extracted_Images/"+filename,"-t","6","-s",str(image.shape[0]), str(image.shape[1]),"-o", folderName+"Segment_Images/jseg/"+"Jseg_"+filename, "1", "-l", "10" ])
    
    def color_segmented_folder(self, folderName):
        """
            Coloração da pasta contendo as imagens segmentas 
        """
        for filename in os.listdir(folderName+"Segment_Images/jseg"):
            if filename.lower().endswith('.JPG') or filename.lower().endswith('.PNG') :
                print(filename)
                jseg_image = Image.open(folderName+"Segment_Images/jseg/"+filename)
                extracted_image = Image.open(folderName+"/Extracted_Images/"+filename.split("Jseg_")[1])
                image = self.remover_linha_branca(self.remover_linha_branca(self.achar_area(self.binarizar_imagem(jseg_image), extracted_image)))
                image.save(folderName + "Segment_Images/jseg/colorida/sem_linha/" + filename)

def main():
    image = Image.open(sys.argv[1])
    print(image)
    scaleFactor = sys.argv[2]
    call(["wine","segwin.exe","-i", sys.argv[1],"-t","6","-s",str(image.size[0]), str(image.size[1]),"-o", scaleFactor+sys.argv[1], "1", "-l", scaleFactor ])

if __name__ == "__main__":    
    main()
