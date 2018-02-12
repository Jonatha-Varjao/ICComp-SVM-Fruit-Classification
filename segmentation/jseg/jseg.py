# -*- coding: utf-8 -*-
import cv2
import os
import sys
from subprocess import call
from PIL import Image, ImageDraw

'''
 Script para utilizar o binário da segmentação do JSEG.
    Utilizando wine 2.0.4 ( para executar binário para windows )
    Scrip ira chamar o programa na qual faz a segmentação utilizando Jseg
    será segmentadas as imagens que já estão previamente pré-processadas
'''


class Jseg:
    # Segmentação da pasta contendo as imagens pre processadas
    def segment_folder(self, folderName):
        for filename in os.listdir(folderName+"Extracted_Images/"):
            if '.' in filename:
                print(filename)
                image = cv2.imread(folderName+"Extracted_Images/"+filename)
                print(str(image.shape[0]),image.shape[1])
                call(["wine","jseg/segwin.exe","-i",folderName+"Extracted_Images/"+filename,"-t","6","-s",str(image.shape[0]), str(image.shape[1]),"-o", folderName+"Segment_Images/jseg/"+"Jseg_"+filename, "1", "-l", "10" ])

'''
FIX: AUMENTAR A ÁREA DA IMAGEM CROPADA PARA MELHORAR A SEGMENTAÇÃO USANDO JSEG [FEITO]
FIX: TESTAR COM SCALES DIFERENES NA HORA DE SEGMENTA:
        -> 100: - NA HORA DA ROTULAÇÃO AS AREAS SE COLIDIRAM 
                - AREAS ESCURAS NA IMAGEM SEGMENTADA 
        -> 50: ?
        -> 30: ?
        -> 10: ?
    #TODO EROSAO / DILATACAO NA HORA DA ROTULACAO (IMAGEM BINARIZADA)
'''
if __name__ == "__main__":
    
    image = Image.open(sys.argv[1])
    print(image)
    scaleFactor = sys.argv[2]

    call(["wine","segwin.exe","-i", sys.argv[1],"-t","6","-s",str(image.size[0]), str(image.size[1]),"-o", scaleFactor+sys.argv[1], "1", "-l", scaleFactor ])
