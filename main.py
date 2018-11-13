# -*- coding: utf-8 -*-
from segmentation.extraction.extraction import Extract
from segmentation.kmeans.kmeans import Segment
from segmentation.jseg.jseg import Jseg
from segmentation.rotulacao.rotulacao import Rotulacao
from classification.classifier import SVM

def main():
    """ Paths da base de dados """
    manga            = 'database/Manga/'
    macaRoyal        = 'database/MacaRoyal/'
    macaVerde        = 'database/MacaVerde/'
    laranja          = 'database/LaranjaTangerina/'
    laranjaInfectada = 'database/LaranjaInfectada/'
    jseg_sem_linha   = 'Segment_Images/jseg/colorida/sem_linha/'
    kmeans           = 'Segment_Images/kmeans/'

    """ Criação dos objetos """
    #seg              = Segment()
    #extraction       = Extract()
    #jseg             = Jseg()
    #colorir          = Rotulacao()
    
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

    """ PROCESSAMENTO DAS IMAGENS (K-MEANS) """
    #seg.segment_folder(manga)
    #seg.segment_folder(laranja)
    #seg.segment_folder(laranjaInfectada)
    #seg.segment_folder(macaRoyal)
    #seg.segment_folder(macaVerde)

    """ POS PROCESSAMENTO DAS IMAGENS ( JSEG ) """
    # COLORAÇÃO DAS AREAS
    #jseg.color_segmented_folder(laranjaInfectada)
    #jseg.color_segmented_folder(laranja)
    #jseg.color_segmented_folder(macaRoyal)
    #jseg.color_segmented_folder(macaVerde)
    #jseg.color_segmented_folder(manga)

    """ POS PROCESSAMENTO DAS IMAGENS ( gPb ) """
    #colorir.extract_folder(macaRoyal, 2)
    #colorir.extract_folder(macaVerde, 2)
    #colorir.extract_folder(manga, 2)
    #colorir.extract_folder(laranjaInfectada, 2)
    #colorir.extract_folder(laranja, 2)

    """ POS PROCESSAMENTO DAS IMAGENS (K-MEANS) """

    """ CLASSIFICACAO DAS IMAGENS """
    
    #  CLASSIFICADORES TREINADOS COM ESPAÇO DE COR RGB
    # svm_linear_rgb  = SVM(nome="SVM_Linear_RGB",   kernel='linear',     cor_space='RGB')
    # svm_rbf_rgb     = SVM(nome="SVM_RBF_RGB",      kernel='rbf',        cor_space='RGB')
    # svm_poly_rgb    = SVM(nome="SVM_POLY_RGB",     kernel='polynomial', cor_space='RGB')
    # svm_sigmoid_rgb = SVM(nome="SVM_SIGMOID_RGB",  kernel='sigmoid',    cor_space='RGB')
    
    # # CLASSIFICADORES TREINADOS COM ESPAÇO DE COR HSV
    # svm_linear_hsv  = SVM(nome="SVM_Linear_HSV",   kernel='linear',    cor_space='HSV')
    # svm_rbf_hsv     = SVM(nome="SVM_RBF_HSV",      kernel='rbf',       cor_space='HSV')
    # svm_poly_hsv    = SVM(nome="SVM_POLY_HSV",     kernel='poly',      cor_space='HSV')
    # svm_sigmoid_hsv = SVM(nome="SVM_SIGMOID_HSV",  kernel='sigmoid',   cor_space='HSV')


    #svm.train_data(laranja+jseg_sem_linha, laranjaInfectada+jseg_sem_linha, 'linear')
    #svm.train_data(laranja+kmeans, laranjaInfectada+kmeans, 'linear')

if __name__ == "__main__":
    main()