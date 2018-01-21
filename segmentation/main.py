# -*- coding: utf-8 -*-
from extraction.extraction import * 
from kmeans.kmeans import * 
from jseg.jseg import *


'''
    "Main" dos scripts
'''

if __name__ == "__main__":
    # HARD-CODE dos paths das imagens
    # Trocar o path para a máquina local
    manga            = '/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/Manga/'
    macaRoyal        = '/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/MacaRoyal/'
    macaVerde        = '/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/MacaVerde/'
    laranja          = '/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/LaranjaTangerina/'
    laranjaInfectada = '/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/LaranjaInfectada/'

    seg              = Segment()
    extraction       = Extract()
    jseg             = Jseg()
    
    # Teste individual com alguma imagem do database
    #image = extraction.crop_image(extraction.extract_countor('/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/LaranjaInfectada/2013_10_25__0063.jpg'))
    #cv2.imwrite("teste.jpg", image)
    
    # PRE-PROCESSAMENTO
    # Extração das frutas nas imagens
    #extraction.extract_folder(manga)
    #extraction.extract_folder(laranja)
    #extraction.extract_folder(laranjaInfectada)
    #extraction.extract_folder(macaRoyal)
    #extraction.extract_folder(macaVerde)

    
    # PROCESSAMENTO DAS IMAGENS (JSEG)
    jseg.segment_folder(macaRoyal)
    
    
    # PROCESSAMENTO DAS IMAGENS (K-MEANS)
    #seg.segment_folder(manga)
    #seg.segment_folder(laranja)
    #seg.segment_folder(laranjaInfectada)

