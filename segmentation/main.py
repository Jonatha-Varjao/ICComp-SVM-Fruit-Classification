# -*- coding: utf-8 -*-
from extraction.extraction import * 
from kmeans.kmeans import * 


'''
    "Main" dos scripts
'''

if __name__ == "__main__":
    # HARD-CODE dos paths das imagens
    # Trocar o path para a máquina local
    manga            = '/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/Manga/Extracted_Images/'
    laranja          = '/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/LaranjaTangerina/Extracted_Images/'
    laranjaInfectada = '/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/LaranjaInfectada/Extracted_Images'

    #seg              = Segment()
    Extraction       = Extract()
    
    # PRE-PROCESSAMENTO
    # Extração das frutas nas imagens
    #Extraction.extract_folder(manga)
    #Extraction.extract_folder(laranja)
    #Extraction.extract_folder(laranjaInfectada)
    
    # PROCESSAMENTO DAS IMAGES (K-MEANS)
    #seg.segmentFolder(manga)
    #seg.segmentFolder(laranja)
    #seg.segmentFolder(laranjaInfectada)

