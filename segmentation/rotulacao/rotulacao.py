# -*- coding: utf-8 -*-
# LABELLING METHOD USING UNION-FIND ARRAYS
from PIL import Image, ImageDraw
import sys
from segmentation.rotulacao.ufarray import UFarray
import os
import cv2
import numpy as np

"""
    Class Rotualação
        Recebe uma imagem segmentada (jseg/gPb) e faz a redução de cores em cada área segmentada pelo
    algoritmo.
        TODO: - Traçar um limiar das áreas "podres"
"""
class Rotulacao:
    
    def binarizar_imagem(self, img):
        """
            Binarizacao da imagem utilizando treshould de 180
        """
        new_image = Image.new("RGB", (img.size[0], img.size[1]))
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                pixel = img.getpixel((i, j))
                media = (pixel[0] + pixel[1] + pixel[2])/3
                if media > 180:
                    new_image.putpixel((i, j), (255, 255, 255))
                else:
                    new_image.putpixel((i, j), (0, 0, 0))
        # erosao
        erosionImage = cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)
        kernel = np.ones((2,2),np.uint8)
        dilatation = cv2.dilate(erosionImage, kernel, iterations = 1)
        new_image = Image.fromarray(dilatation)
        return new_image
    
    def cor_area(self, pixel_Labeados, RGB_Label):
        """
            Média de cor da área segmentada
        """
        somaR = 0
        somaG = 0
        somaB = 0
        NovasCores = []
        for i in range(1, len(pixel_Labeados)):
            if pixel_Labeados[i-1][1] == pixel_Labeados[i][1]:
                somaR = somaR + pixel_Labeados[i-1][0][0] + pixel_Labeados[i][0][0]
                somaG = somaG + pixel_Labeados[i-1][0][1] + pixel_Labeados[i][0][1]
                somaB = somaB + pixel_Labeados[i-1][0][2] + pixel_Labeados[i][0][2]
                #print("Somou: "+str(somaR)+" "+str(somaG)+" "+str(somaB))
            else:
                somaR = somaR / RGB_Label.count(pixel_Labeados[i-1][1])
                somaG = somaG / RGB_Label.count(pixel_Labeados[i-1][1])
                somaB = somaB / RGB_Label.count(pixel_Labeados[i-1][1])
                NovasCores.append((somaR, somaG, somaB, pixel_Labeados[i-1][1]))
                somaR, somaG, somaB = 0, 0, 0
                #print("Soma Zerou: "+str(somaR)+" "+str(somaG)+" "+str(somaB))
            if i == len(pixel_Labeados)-1:
                somaR = somaR / RGB_Label.count(pixel_Labeados[i][1])
                somaG = somaG / RGB_Label.count(pixel_Labeados[i][1])
                somaB = somaB / RGB_Label.count(pixel_Labeados[i][1])
                NovasCores.append((somaR, somaG, somaB, pixel_Labeados[i][1]))
                somaR, somaG, somaB = 0, 0, 0
                #print("Ultima soma: "+str(somaR)+" "+str(somaG)+" "+str(somaB))
        return NovasCores

    def achar_area(self, img, imgOriginal):
        """
            Encontro as áreas da segmentação da imagem e as pinto
            conforme sua cor
        """
        img = img.convert('L')
        matrizPixel = img.load()
        width, height = img.size
        # Instacia da minha estrutura UNION FIND / dicionario das labels
        rotulacao = Rotulacao()
        ufFlatten = UFarray()
        uf = UFarray()
        labels = {}
        colors = {}
        # Elimino as equivalencias
        ufFlatten = uf.rotulacao_pixel(labels, height, width, matrizPixel)
        # Crio nova imagem e uma matriz dessa imagem
        saida_img = Image.new("RGB", (width, height))
        outmatrizPixel = saida_img.load()
        #Criação do dicionário contendo RGB, REGIÃO
        RGB = []
        RGB_Label = []
        component = []
        pixel_Labeados = []
        # LOOP para armazenar o pixel x,y e sua respectiva label
        for (x, y) in labels:
            # Busco o label da região em que o ponto atual pertence
            RGB.append(imgOriginal.getpixel((x, y)))
            RGB_Label.append(uf.find(labels[(x, y)]))
            labels[(x, y)] = uf.find(labels[(x, y)])
        # OTIMIZAR ESSA LIST COMPREHESION    
        pixel_labeados = list(zip(RGB, RGB_Label))
        pixel_labeados.sort(key=lambda tup: tup[1])
        # Array contendo as novas cores das areas rotuladas
        NovasCores = rotulacao.cor_area(pixel_labeados, RGB_Label)
        outmatrizPixel = rotulacao.pintar_area(NovasCores, labels, outmatrizPixel)
        return saida_img
        
    def pintar_area(self, NovasCores, labels, outmatrizPixel):
        """
            Pinto a área com a média da cor encontrada
        """
        for (x, y) in labels:
            for j in range(len(NovasCores)):
                if labels[(x, y)] == NovasCores[j][3]:
                    red = int(NovasCores[j][0])
                    green = int(NovasCores[j][1])
                    blue = int(NovasCores[j][2])
                    # forçar o background a ser preto
                    if red < 5 and green < 5 and blue < 5:
                        red,green,blue = 0,0,0
                    outmatrizPixel[x, y] = (red, green, blue)
        return outmatrizPixel
    
    # TODO:  ANALISO MINHA 8-VIZINHANCA
    #        SE TENHO 5 PIXEIS VIZINHOS > 5, PEGO O DA DIREITA NAO PRETO E VOU ANALISANDO MINHA VIZINHANCA
    def remover_linha_branca(self, img):
        new_image = img.copy()
        qtd_Pixels = 0
        for i in range(1, img.size[0]-1):
            for j in range(1, img.size[1]-1):
                p1 = img.getpixel((i-1, j-1))
                p2 = img.getpixel((i-1, j))
                p3 = img.getpixel((i-1, j+1))
                p4 = img.getpixel((i, j-1))
                p5 = img.getpixel((i, j))
                p6 = img.getpixel((i, j+1))
                p7 = img.getpixel((i+1, j-1))
                p8 = img.getpixel((i+1, j))
                p9 = img.getpixel((i+1, j+1))
                if ((p5[0] + p5[1] + p5[2])/3) < 5:
                    if ((p1[0] + p1[1] + p1[2])/3) > 5:
                        qtd_Pixels += 1
                    if ((p2[0] + p2[1] + p2[2])/3) > 5:
                        qtd_Pixels += 1
                    if ((p3[0] + p3[1] + p3[2])/3) > 5:
                        qtd_Pixels += 1
                    if ((p4[0] + p4[1] + p4[2])/3) > 5:
                        qtd_Pixels += 1
                    if ((p6[0] + p6[1] + p6[2])/3) > 5:
                        qtd_Pixels += 1
                    if ((p7[0] + p7[1] + p7[2])/3) > 5:
                        qtd_Pixels += 1
                    if ((p8[0] + p8[1] + p8[2])/3) > 5:
                        qtd_Pixels += 1
                    if ((p9[0] + p9[1] + p9[2])/3) > 5:
                        qtd_Pixels += 1
                    #print(qtd_Pixels)
                    if qtd_Pixels >= 3:
                        #print("pixel preto na area colorida")
                        #pixel direita
                        if ((p6[0] + p6[1] + p6[2]) / 3) > 5:
                            new_image.putpixel((i,j), img.getpixel((i, j+1)))
                        elif ((p7[0] + p7[1] + p7[2])/3) > 5 :
                            new_image.putpixel((i,j), img.getpixel((i+1, j-1)))
                        elif ((p8[0] + p8[1] + p8[2])/3) > 5:
                            new_image.putpixel((i,j), img.getpixel((i+1, j)))
                        elif ((p9[0] + p9[1] + p9[2])/3) > 5:
                            new_image.putpixel((i,j), img.getpixel((i+1, j+1)))
                        elif ((p1[0] + p1[1] + p1[2])/3) > 5:
                            new_image.putpixel((i,j), img.getpixel((i-1, j-1)))
                        elif ((p2[0] + p2[1] + p2[2])/3) > 5:
                            new_image.putpixel((i,j), img.getpixel((i-1, j)))
                        elif ((p3[0] + p3[1] + p3[2])/3) > 5:
                            new_image.putpixel((i,j), img.getpixel((i-1, j+1)))
                        qtd_Pixels = 0
                    else:
                        qtd_Pixels = 0
        return new_image
    
        
if __name__ == "__main__": 
    obj = Rotulacao()
    imgOriginal = Image.open(sys.argv[1])
    #img.show()
    #img_teste = obj.remover_linha_branca(obj.achar_area(obj.binarizar_imagem(imgOriginal), imgOriginal))
    #img_teste = obj.remover_linha_branca(img_teste)
    #img_teste = obj.remover_linha_branca(img_teste)
    #img_teste.show()
