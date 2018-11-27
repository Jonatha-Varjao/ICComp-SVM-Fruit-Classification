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

    def __repr__(self):
        return f'{self.mask_x}, {self.mask_y}'

    def feature_vector(self)->List[float]:
        data = deque()
        feature_limiter = 0
        for x in range(0, self.image.size[0] - self.mask_x, self.mask_x):
            for y in range(0, self.image.size[1], self.mask_y):
                if feature_limiter < self.size:
                    data.append(self.median_filter_rgb(x,y))
                    feature_limiter = feature_limiter + 1
                else:
                    break
        print(data)
        print(len(data))
        return data   

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
        median = (r+g+b)/qtdPixel*3
        return median
    
    def median_filter_hsv(self, x, y)->float:
        pass

def main():
    filtro = Filter(Image.open(sys.argv[1]),15)
    filtro.feature_vector()
    
"""
TODO:
    - Testar pegando os pixels pretos produzidos na extração
    - Descartar os pixels pretos na hora da criação do vetor de característica
    - median_filter_hsv 
""" 

if __name__ == '__main__':
    main()
