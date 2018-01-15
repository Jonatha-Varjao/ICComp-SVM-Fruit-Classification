# -*- coding: utf-8 -*-
from kmeans import *
from extraction import Extract

'''
    "Main" dos scripts
'''

if __name__ == "__main__":
    # HARD-CODE dos paths das imagens
    # Trocar o path para a máquina local
    manga            = '/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/Manga/'
    laranja          = '/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/LaranjaTangerina/'
    laranjaInfectada = '/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/LaranjaInfectada/'

    seg              = Segment()
    Extraction       = Extract()
    
    #Extraction.crop_image(Extraction.extract_countor('/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/segmentation/k-means/Orange-ICC_001.JPG'))
    
    Extraction.extract_folder(manga)
    
    #seg.segmentFolder(manga)
    #seg.segmentFolder(laranja)
    #seg.segmentFolder(laranjaInfectada)