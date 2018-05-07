# -*- coding: utf-8 -*-
from extraction.extraction import Extract
from kmeans.kmeans import Segment
from jseg.jseg import Jseg
from rotulacao.rotulacao import Rotulacao

"""
    Script Principal
"""

if __name__ == "__main__":
    """ Strings da base de dados """
    manga            = '../database/Manga/'
    macaRoyal        = '../database/MacaRoyal/'
    macaVerde        = '../database/MacaVerde/'
    laranja          = '../database/LaranjaTangerina/'
    laranjaInfectada = '../database/LaranjaInfectada/'
    
    """ Criação dos objetos """
    seg              = Segment()
    extraction       = Extract()
    jseg             = Jseg()
    colorir          = Rotulacao()
    
    """ PRE-PROCESSAMENTO ( CROP DO BACKGROUND DA FRUTA ) """
    # Extração das frutas nas imagens
    # extraction.extract_folder(manga)
    # extraction.extract_folder(laranja)
    # extraction.extract_folder(laranjaInfectada)
    # extraction.extract_folder(macaRoyal)
    # extraction.extract_folder(macaVerde)

    """ PROCESSAMENTO DAS IMAGENS (JSEG) """
    #jseg.segment_folder(macaRoyal)
    #jseg.segment_folder(laranjaInfectada)
    #jseg.segment_folder(laranja)
    #jseg.segment_folder(manga)
    #jseg.segment_folder(macaVerde)

    """ POS PROCESSAMENTO DAS IMAGENS ( JSEG ) """
    # COLORAÇÃO DAS AREAS
    #jseg.extract_folder(laranjaInfectada)
    #jseg.extract_folder(laranja)
    #jseg.extract_folder(macaRoyal)
    #jseg.extract_folder(macaVerde)
    #jseg.extract_folder(manga)

    """ POS PROCESSAMENTO DAS IMAGENS ( gPb ) """
    #colorir.extract_folder(macaRoyal, 2)
    #colorir.extract_folder(macaVerde, 2)
    #colorir.extract_folder(manga, 2)
    #colorir.extract_folder(laranjaInfectada, 2)
    #colorir.extract_folder(laranja, 2)

    """ PROCESSAMENTO DAS IMAGENS (K-MEANS) """
    #seg.segment_folder(manga)
    #seg.segment_folder(laranja)
    #seg.segment_folder(laranjaInfectada)
    #seg.segment_folder(macaRoyal)
    #seg.segment_folder(macaVerde)

