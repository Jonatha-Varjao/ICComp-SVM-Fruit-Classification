# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

'''
    Implementação para extração de objetos de um dado plano de fundo.
    Inspiração: https://gist.github.com/Munawwar/0efcacfb43827ba3a6bac3356315c419 
                https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
'''

class Extract:

    def crop_image(self, img):
        
        mask = img > 0

        # Coordinates of non-black pixels.
        coords = np.argwhere(mask)

        # Bounding box of non-black pixels.
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top

        # Get the contents of the bounding box.
        cropped = img[x0:x1, y0:y1, z0:z1]
        return cropped


    def getSobel (self,channel):

        sobelx = cv2.Sobel(channel, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE)
        sobely = cv2.Sobel(channel, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE)
        sobel = np.hypot(sobelx, sobely)

        return sobel;

    def findSignificantContours (self,img, sobel_8u):
        image, contours, heirarchy = cv2.findContours(sobel_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find level 1 contours
        level1 = []
        for i, tupl in enumerate(heirarchy[0]):
            # Each array is in format (Next, Prev, First child, Parent)
            # Filter the ones without parent
            if tupl[3] == -1:
                tupl = np.insert(tupl, 0, [i])
                level1.append(tupl)

        # From among them, find the contours with large surface area.
        significant = []
        tooSmall = sobel_8u.size * 10 / 100 # If contour isn't covering 5% of total area of image then it probably is too small
        for tupl in level1:
            contour = contours[tupl[0]];
            area = cv2.contourArea(contour)
            if area > tooSmall:
                cv2.drawContours(img, [contour], 0, (0,0,0),2, cv2.LINE_AA, maxLevel=1)
                significant.append([contour, area])
        
        significant.sort(key=lambda x: x[1])
        return [x[0] for x in significant];

    def extract_countor (self, path):
        img = cv2.imread(path)
        blurred = cv2.GaussianBlur(img, (3, 3), 0) # Filtro Gaussiano
        # Operador Sobel
        sobel = np.max( np.array([ self.getSobel(blurred[:,:, 0]), self.getSobel(blurred[:,:, 1]), self.getSobel(blurred[:,:, 2]) ]), axis=0 )
        mean = np.mean(sobel)
        sobel[sobel <= mean] = 0;
        sobel[sobel > 255] = 255;
        # Imagem apos o filtro de sobel
        #cv2.imwrite('edge.png', sobel);
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
        img[mask] = 0;
        fname = path.split('/')[-1]
        #cv2.imwrite('output' + fname, img)
        #print (path)
        return img
    
    def extract_folder(self, folderName):
        extraction = Extract()
        for filename in os.listdir(folderName):
            if '.' in filename:
                print(filename)
                image = extraction.crop_image(extraction.extract_countor(folderName+filename))
                cv2.imwrite(folderName + "Extracted_Images/" + filename, image)
                #label,result= seg.kmeans(image)
                #cv2.imshow("segmenteda",result)
                #cv2.imwrite(folderName + "Segment_Images/kmeans/Kmeans_" + filename, result)
                # for cluster in range(seg.segments):
                #     result=seg.extractComponent(image, label, cluster)
                #     cv2.imwrite("Component_"+str(cluster)+"_"+imageString[0], result)
        return image


#TODO: Criar uma classe e aglomerar as funções [FEITO]
#TODO: Criar método para acessar um path, e extrair o objeto dela
