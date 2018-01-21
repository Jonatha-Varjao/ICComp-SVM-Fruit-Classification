# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys
import os
from subprocess import call

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
                image = cv2.imread(folderName+filename)
                print(str(image.shape[0]),image.shape[1])
                call(["wine","jseg/segwin.exe","-i",folderName+filename,"-t","6","-s",str(image.shape[0]), str(image.shape[1]),"-o", folderName+"Segment_Images/jseg/"+"Jseg_"+filename, "1"])
        pass

# FIX: AUMENTAR A ÁREA DA IMAGEM CROPADA PARA MELHORAR A SEGMENTAÇÃO USANDO JSEG [FEITO]

if __name__ == "__main__":
    call(["pwd",""])