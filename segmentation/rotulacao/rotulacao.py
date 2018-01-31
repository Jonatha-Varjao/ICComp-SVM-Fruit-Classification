# -*- coding: utf-8 -*-

# LABELLING METHOD USING UNION-FIND ARRAYS
from PIL import Image, ImageDraw

import sys
from itertools import product
from rotulacao.ufarray import UFarray
import os
import cv2


class Rotulacao:
    
    def binarizar_imagem(self, img):
        new_image = Image.new("RGB",(img.size[0],img.size[1]))
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                p = img.getpixel((i,j))
                m = (p[0] + p[1] + p[2])/3
                if m > 180:
                    new_image.putpixel((i,j),(255,255,255))
                else:
                    new_image.putpixel((i,j),(0,0,0))
        return new_image

    def achar_pintar_area(self, img, imgOriginal):
        img = img.convert('L')
        matrizPixel = img.load()
        width, height = img.size
    
        # Instacia da minha estrutura UNION FIND / dicionario das labels
        uf = UFarray()
        labels = {}
        colors = {}
    
        for y, x in product(range(height), range(width)):
    
            #
            # Condicoes de vizinhas dos meu pixeis:
            #
            #   -------------
            #   | a | b | c |
            #   -------------
            #   | d | e |   |
            #   -------------
            #   |   |   |   |
            #   -------------
            #
            # Se o meu pixel for 'e'
            # a, b, c, e d sao meus vizinhos de interesse
            # 255 branco, 0 = preto
            # pixeis brancos sao ignorados
    
            # Pixel branco, ignoro
            if matrizPixel[x, y] == 255:
                pass
    
            # Se o pixel b for preto :
            # a,c,e sao seus vizinhos, logo fazem parte da mesma regiao
            # e como e é vizinho de d, assumo que b = e = d
            
            elif y > 0 and matrizPixel[x, y-1] == 0:
                labels[x, y] = labels[(x, y-1)]
    
            # Se o pixel c for preto :
            #    b é seu vizinho, mas a e d não
            #    logo checo a label de 'a' e 'd'
            elif x+1 < width and y > 0 and matrizPixel[x+1, y-1] == 0:
    
                c = labels[(x+1, y-1)]
                labels[x, y] = c
    
                # Se a for petro:
                #    logo a e c estão na mesma regiao
                #    adiciono na união (c,a)
                if x > 0 and matrizPixel[x-1, y-1] == 0:
                    a = labels[(x-1, y-1)]
                    uf.union(c, a)
    
                # Se d for petro:
                #    logo d e c estão na mesma região
                #    adiciona na união (c,d)
                elif x > 0 and matrizPixel[x-1, y] == 0:
                    d = labels[(x-1, y)]
                    uf.union(c, d)
    
            # Se a for preto:
            #    sabemos que c e b sao brancos
            #    d is a's neighbor, so they already have the same label
            #    lgoo seto a label a em e
            elif x > 0 and y > 0 and matrizPixel[x-1, y-1] == 0:
                labels[x, y] = labels[(x-1, y-1)]
    
            # Se o d for preto:
            #    logo a,b,c são brancos
            #    logo seto a label d em e
            elif x > 0 and matrizPixel[x-1, y] == 0:
                labels[x, y] = labels[(x-1, y)]
    
            # toda minha vizinhança é branca
            # logo o pixel atual recebe uma nova label
            else: 
                labels[x, y] = uf.criaLabel()
    
    
        # Elimino as equivalencias
        uf.flatten()
        

        # Crio nova imagem e uma matriz dessa imagem
        saida_img = Image.new("RGB", (width, height))
        outmatrizPixel = saida_img.load()
        
        #Criação do dicionário contendo RGB, REGIÃO
        RGB             = []
        RGB_Label       = []
        component       = []
        pixel_Labeados  = []
        
        # LOOP para computar a media dos pixels internos
        for (x, y) in labels:
            # Busco o label da região em que o ponto atual pertence
            RGB.append(imgOriginal.getpixel((x,y)))
            RGB_Label.append(uf.find(labels[(x, y)]))
            labels[(x, y)] = uf.find(labels[(x, y)])
            #print(labels[(x, y)])
            #component.append(tuplaRGB_Label)
            #print(component)
            # Associo cor random com a regiao
            # if component not in colors: 
            #     colors[component] = (random.randint(0,255), random.randint(0,255),random.randint(0,255))
            #Pinto as regioes
            #outmatrizPixel[x, y] = colors[component]
        #print(RGB)
        pixel_Labeados = list(zip(RGB , RGB_Label))
        pixel_Labeados.sort(key=lambda tup: tup[1])
        
        #print(pixel_Labeados)
        #print(len(pixel_Labeados))
        somaR    = 0
        somaG    = 0
        somaB    = 0
        NovasCores = []
        Conj_Labels = list(set(uf.P))
        
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
                NovasCores.append((somaR, somaG, somaB, pixel_Labeados[i-1][1] ))
                somaR, somaG, somaB = 0,0,0
                #print("Soma Zerou: "+str(somaR)+" "+str(somaG)+" "+str(somaB))
            if i == len(pixel_Labeados)-1:
                somaR = somaR / RGB_Label.count(pixel_Labeados[i][1])
                somaG = somaG / RGB_Label.count(pixel_Labeados[i][1])
                somaB = somaB / RGB_Label.count(pixel_Labeados[i][1])
                NovasCores.append((somaR, somaG, somaB, pixel_Labeados[i][1] ))
                somaR, somaG, somaB = 0,0,0
                #print("Ultima soma: "+str(somaR)+" "+str(somaG)+" "+str(somaB))
        
        #print(NovasCores)  

        #print(uf.P)

        dicionario_pixel_Labeados = dict(pixel_Labeados)
        #print(dicionario_pixel_Labeados)


        for (x, y) in labels:
            for j in range(len(NovasCores)):
                if labels[(x, y)] == NovasCores[j][3]:
                    R = int(NovasCores[j][0])
                    G = int(NovasCores[j][1])
                    B = int(NovasCores[j][2])
                    #print(R,G,B)
                    outmatrizPixel[x, y] = (R,G,B)


        #print(sorted(set(uf.P)))
        #print("Regioes:" + str(len(set(uf.P))))
        
        return saida_img
    
    # ANALISO MINHA 8-VIZINHANCA
    # SE TENHO 5 PIXEIS VIZINHOS > 5, PEGO O DA DIREITA SENÃO FOR PRETO E VOU ANALISANDO MINHA 8-VIZINHANCA

    def remover_linha_branca(self, img):
        new_image = img.copy()
        qtd_Pixels = 0
        for i in range(1, img.size[0]-1):
            for j in range(1,img.size[1]-1):
                p1 = img.getpixel((i-1,j-1))
                p2 = img.getpixel((i-1,j))
                p3 = img.getpixel((i-1,j+1))
                p4 = img.getpixel((i,j-1))
                p5 = img.getpixel((i,j))
                p6 = img.getpixel((i,j+1))
                p7 = img.getpixel((i+1,j-1))
                p8 = img.getpixel((i+1,j))
                p9 = img.getpixel((i+1,j+1))
                
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
                            new_image.putpixel((i,j), img.getpixel((i,j+1)))
                        elif ((p7[0] + p7[1] + p7[2])/3) > 5 :
                            new_image.putpixel((i,j), img.getpixel((i+1,j-1)))
                        elif ((p8[0] + p8[1] + p8[2])/3) > 5:
                            new_image.putpixel((i,j), img.getpixel((i+1,j)))
                        elif ((p9[0] + p9[1] + p9[2])/3) > 5:
                            new_image.putpixel((i,j), img.getpixel((i+1,j+1)))
                        elif ((p1[0] + p1[1] + p1[2])/3) > 5:
                            new_image.putpixel((i,j), img.getpixel((i-1,j-1)))
                        elif ((p2[0] + p2[1] + p2[2])/3) > 5:
                            new_image.putpixel((i,j), img.getpixel((i-1,j)))
                        elif ((p3[0] + p3[1] + p3[2])/3) > 5:
                            new_image.putpixel((i,j), img.getpixel((i-1,j+1)))
                        
                        qtd_Pixels = 0
                    
                    else:
                        qtd_Pixels = 0
        
        return new_image
    
    # EXTRACAO FOLDER(1 - JSEG , 2 - GPB)
    def extract_folder(self, folderName, metodo):
        if metodo == 1:
            for filename in os.listdir(folderName+"Segment_Images/jseg"):
                if '.' in filename:
                    print(filename)
                    imgOriginal = Image.open(folderName+"Segment_Images/jseg/"+filename)
                    image = self.remover_linha_branca(self.achar_pintar_area(self.binarizar_imagem(imgOriginal), imgOriginal))
                    image.save(folderName + "Segment_Images/jseg/colorida/" + filename)
        elif metodo == 2:
            for filename in os.listdir(folderName+"Segment_Images/gPb"):
                if '.' in filename:
                    print(filename)
                    imgOriginal = Image.open(folderName+"Segment_Images/gPb/"+filename)
                    image = self.remover_linha_branca(self.achar_pintar_area(self.binarizar_imagem(imgOriginal), imgOriginal))
                    image.save(folderName + "Segment_Images/gPb/colorida/" + filename)
        
        
def main():
    
    obj = Rotulacao()
    imgOriginal = Image.open(sys.argv[1])
    #img.show()
    img_teste = obj.remover_linha_branca(obj.achar_pintar_area(obj.binarizar_imagem(imgOriginal), imgOriginal))
    img_teste.show()
    #img.show()
    #img.save("Binarizada"+sys.argv[1])
    
if __name__ == "__main__": main()