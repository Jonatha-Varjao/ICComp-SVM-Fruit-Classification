# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

'''
    Implementação para extração de objetos de um dado plano de fundo homogêno [PRETO].
    Inspiração: https://gist.github.com/Munawwar/0efcacfb43827ba3a6bac3356315c419 
                https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
'''
class Extract:

    def crop_image(self, img):
        """
            Cropo a imagem retirando o fruto.
        """
        maskBG = img > 0
        coords = np.argwhere(maskBG)
        # Caixa de valores de pixels não-pretos.
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1   # fatias são exclusivas no topo
        # Obter o conteúdo da caixa delimitadora.
        print(x0,x1,y0,y1,z0,z1)
        
        # Aumentado um pouco a caixa delimitadora ( devido ao jseg nao fechar as regiões em algumas imagens )
        x0_maior = x0 - 25
        x1_maior = x1 + 25
        y0_maior = y0 - 25
        y1_maior = y1 + 25
        
        if x0_maior < 0:
            x0_maior = 0
        if y0_maior < 0:
            y0_maior = 0
        
        print(img.shape)
        print(x0,x1,y0,y1,z0,z1)
        cropped = img[x0_maior:x1_maior, y0_maior:y1_maior, z0:z1]
        return cropped

    def getSobel(self,channel):
        sobelx = cv2.Sobel(channel, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE)
        sobely = cv2.Sobel(channel, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE)
        sobel = np.hypot(sobelx, sobely)
        return sobel

    def findSignificantContours(self, img, sobel_8u):
        image, contours, heirarchy = cv2.findContours(sobel_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Encontre contornos de nível 1
        level1 = []
        for i, tupl in enumerate(heirarchy[0]):
            # Cada matriz está no formato (Next, Prev, First child, Parent)
            # Filtre os sem pai
            if tupl[3] == -1:
                tupl = np.insert(tupl, 0, [i])
                level1.append(tupl)

        # Entre eles, encontre os contornos com grande área de superfície.
        significant = []
        tooSmall = sobel_8u.size * 10 / 100 # Se o contorno não cobre 5% da área total da imagem, provavelmente é muito pequeno
        for tupl in level1:
            contour = contours[tupl[0]]
            area = cv2.contourArea(contour)
            if area > tooSmall:
                cv2.drawContours(img, [contour], 0, (0,0,0),2, cv2.LINE_AA, maxLevel=1)
                significant.append([contour, area])
        
        significant.sort(key=lambda x: x[1])
        return [x[0] for x in significant]

    def extract_countor(self, path):
        img = cv2.imread(path)
        blurred = cv2.GaussianBlur(img, (3, 3), 0) # Filtro Gaussiano
        # Operador Sobel
        sobel = np.max( np.array([ self.getSobel(blurred[:,:, 0]), self.getSobel(blurred[:,:, 1]), self.getSobel(blurred[:,:, 2]) ]), axis=0 )
        mean = np.mean(sobel)
        sobel[sobel <= mean] = 0
        sobel[sobel > 255] = 255
        # Imagem apos o filtro de sobel
        #cv2.imwrite('edge.png', sobel)
        sobel_8u = np.asarray(sobel, np.uint8)
        # Contornos encontrados
        significant = self.findSignificantContours(img, sobel_8u)
        # Mascara
        mask = sobel.copy()
        mask[mask > 0] = 0
        cv2.fillPoly(mask, significant, 255)
        # Inversão da Máscara
        mask = np.logical_not(mask)
        # Remoção do Background
        imgTeste = img.copy()
        imgTeste[mask] = 0
        # Coordenadas dos pixels pretos
        maskBG = imgTeste > 0
        coords = np.argwhere(maskBG)
        if len(coords) == 0:
            return img
        else:
            return imgTeste
        
    def extract_folder(self, folderName):
        for filename in os.listdir(folderName):
            if '.' in filename:
                print(filename)
                image = self.crop_image(self.extract_countor(folderName+filename))
                cv2.imwrite(folderName + "Extracted_Images/" + filename, image)
        return image
    

#TODO: Criar uma classe e aglomerar as funções [FEITO]
#TODO: Criar método para acessar um path, e extrair o objeto dela [FEITO]
#TODO: FIX no error do imread / imwrite do openCV particularmente com só uma image até agora Orange-ICC_131.JPG
#       - Algumas imagens (Orange-ICC_131.JPG) ñ consigo tirar o contorno usando sobel
#       - Fix: Segmentar sem cropar ela....        
