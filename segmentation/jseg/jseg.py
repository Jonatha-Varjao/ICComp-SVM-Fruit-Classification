# -*- coding: utf-8 -*-
import cv2
import os
import sys
from subprocess import call
from PIL import Image, ImageDraw
from rotulacao.rotulacao import Rotulacao

'''
 Script para utilizar o binário da segmentação do JSEG.
    Utilizando wine 2.0.4 ( para executar binário para windows )
    Scrip ira chamar o programa na qual faz a segmentação utilizando Jseg
    será segmentadas as imagens que já estão previamente pré-processadas
'''


class Jseg(Rotulacao):
    # Segmentação da pasta contendo as imagens pre processadas
    def segment_folder(self, folderName):
        for filename in os.listdir(folderName+"Extracted_Images/"):
            if '.' in filename:
                image = cv2.imread(folderName+"Extracted_Images/"+filename)
                print(filename, str(image.shape[0]),image.shape[1])
                call(["wine","jseg/segwin.exe","-i",folderName+"Extracted_Images/"+filename,"-t","6","-s",str(image.shape[0]), str(image.shape[1]),"-o", folderName+"Segment_Images/jseg/"+"Jseg_"+filename, "1", "-l", "10" ])
    # Coloração da pasta contendo as imagens segmentas 
    def extract_folder(self, folderName):
        for filename in os.listdir(folderName+"Segment_Images/jseg"):
            if '.' in filename:
                print(filename)
                jseg_image = Image.open(folderName+"Segment_Images/jseg/"+filename)
                extracted_image = Image.open(folderName+"/Extracted_Images/"+filename.split("Jseg_")[1])
                image = self.remover_linha_branca(self.achar_area(self.binarizar_imagem(jseg_image), extracted_image))
                image.save(folderName + "Segment_Images/jseg/colorida/" + filename)

'''
FIX: AUMENTAR A ÁREA DA IMAGEM CROPADA PARA MELHORAR A SEGMENTAÇÃO USANDO JSEG [FEITO]
FIX: TESTAR COM SCALES DIFERENES NA HORA DE SEGMENTA:
        -> 100: - NA HORA DA ROTULAÇÃO AS AREAS SE COLIDIRAM 
                - AREAS ESCURAS NA IMAGEM SEGMENTADA 
        -> 50: - IGUAL AO 100
        -> 30: - IGUAL A0 100
        -> 10: - IGUAL AO 100
    #TODO EROSAO / DILATACAO NA HORA DA ROTULACAO (IMAGEM BINARIZADA)
        - RESOLVEU O PROBLEMA DA COLORAÇÃO
        - NA HORA DA ACHAR A NOVA COR TÔ PEGANDO A IMAGEM EXTRAÍDA
'''
if __name__ == "__main__":
    
    image = Image.open(sys.argv[1])
    print(image)
    scaleFactor = sys.argv[2]

    call(["wine","segwin.exe","-i", sys.argv[1],"-t","6","-s",str(image.size[0]), str(image.size[1]),"-o", scaleFactor+sys.argv[1], "1", "-l", scaleFactor ])
