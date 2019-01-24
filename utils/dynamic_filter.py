# -*- coding: utf-8 -*-
import os
import sys

from PIL import Image
from typing import List
from collections import deque


class Filter(object):
    """
        Filtro Dinâmico aplicando a média da máscara gerada.
    """
    
    def __init__(self, image, batch_size=8):
        """
            Construtor da minha classe representando o filtro média dinâmico
        """
        self.batch_size = batch_size
        self.size = batch_size**2
        self.image = image
        self.mask_x, self.mask_y = image.size[0]//batch_size, image.size[1]//batch_size

    def feature_vector(self)->List[float]:
        data = deque()
        feature_limiter = 0
        for x in range(0, self.image.size[0] - self.mask_x, self.mask_x):
            for y in range(0, self.image.size[1], self.mask_y):
                if feature_limiter < self.size:
                    data.append(self.median_filter_hsv(x,y))
                    feature_limiter = feature_limiter + 1
                else:
                    break
        print(data)
        print(len(data))
        return data   

    def rgb_to_hsv(self, r, g, b):
        ridx, gidx, bidx = r/256, g/256, b/256
        Max, Min = max(ridx,gidx,bidx), min(ridx,gidx,bidx)
        Delta = Max-Min
        if Delta == 0:
            Hue = 0
        elif Max == ridx:
            Hue = 60 * ( ((gidx-bidx)/Delta) % 6  ) 
        elif Max == gidx:
            Hue = 60 * ( ((bidx-ridx)/Delta) + 2  ) 
        elif Max == bidx:
            Hue = 60 * ( ((ridx-gidx)/Delta) + 4  ) 
        if Max == 0:
            Saturation = 0
        else:
            Saturation = Delta/Max
        Value = Max
        return Hue, Saturation, Value

    def median_filter_rgb(self, x, y)->float:
        red,green,blue,qtdPixel = 0,0,0,0
        # Altura overflow .....
        if y+self.mask_y > self.image.size[1]:
            dif_y = self.image.size[1] - y
        else:
            dif_y = self.mask_y    
        for i in range(x, x+self.mask_x):
            for j in range(y, y+dif_y):
                r, g, b = self.image.getpixel((i,j))
                red += r
                green += g
                blue += b
                qtdPixel = qtdPixel+1
        # quantized RGB into grayscale 0->1
        median = (red+green+blue)/(qtdPixel*3*255)
        return median
    
    def median_filter_hsv(self, x, y)->float:
        red,green,blue,qtdPixel = 0,0,0,0
        # Altura overflow .....
        if y+self.mask_y > self.image.size[1]:
            dif_y = self.image.size[1] - y
        else:
            dif_y = self.mask_y    
        for i in range(x, x+self.mask_x):
            for j in range(y, y+dif_y):
                r, g, b = self.image.getpixel((i,j))
                red += r
                green += g
                blue += b
                qtdPixel = qtdPixel+1
        red, green, blue = red/qtdPixel, green/qtdPixel, blue/qtdPixel
        H,S,V = self.rgb_to_hsv(red,green,blue)
        return V

def main():
    filtro = Filter(Image.open(sys.argv[1]),3)
    filtro.feature_vector()
    
"""
TODO:
    - Testar pegando os pixels pretos produzidos na extração
    - Descartar os pixels pretos na hora da criação do vetor de característica
    - median_filter_hsv 
""" 

if __name__ == '__main__':
    main()
