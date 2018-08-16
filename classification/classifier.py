# -*- coding: utf-8 -*-
from PIL import Image
from io import StringIO
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
from sklearn.externals import joblib
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time


"""
    Classificador SVM (kernel linear)
"""
class SVM:
    """
        Classe representando classificador SVM
        para uma classificação binária.
    """    
    
    def train_data(self, path_a: str, path_b: str, print_metrics=True):
        """
            Treina o classificador. path_a e path_b devem ser paths de pastas. 
            path_a e path_b são processadas por process_folder().

            Args:
            path_a (str): diretório contendo imagens da classe A.
            path_b (str): diretório contendo imagens da classe B.
            print_metrics  (boolean, optional): se True, printar estatísticas
            sobre a perfomace do classificador.

            Returns:
            Um classificador (sklearn.svm.SVC).
        """
        if not os.path.isdir(path_a):
            raise IOError('%s não é um diretório' % path_a)
        if not os.path.isdir(path_b):
            raise IOError('%s não é um diretório' % path_b)
        training_a = self.process_folder(path_a)
        training_b = self.process_folder(path_b)
        # data contendo todo training set (lista de vetor de features)
        data = training_a + training_b
        # target é a lista das classes para cada vetor de features: 
        # '1' pra classe A e '0' pra classe B
        target = [1] * len(training_a) + [0] * len(training_b)        
        # # divido os dados de treinamento em um set de teste e treinamento
        # o set de teste vai conter 30% do total
        x_train, x_test, y_train, y_test = model_selection.train_test_split(data,
                target, test_size=0.30)
        # plotando o gráfico
        # plot_graph(y_train, y_test)
        # definindo uma busca para os melhores parametros 
        parameters = {'kernel': ['linear'], 'C': [1, 10, 100, 1000],
                'gamma': [0.1, 0.01, 0.001, 0.0001]}
        # procura os melhores parametros para kernel linear variando o C e a gamma
        clf = model_selection.GridSearchCV(svm.SVC(), parameters, scoring='f1_macro', cv=10).fit(x_train, y_train)
        # cross-fold validation prediciont usando 10 folds
        classifier = clf.best_estimator_
        prediciton = classifier.predict(x_test)
        if print_metrics:
            print()
            print('Parameters:', clf.best_params_)
            print()
            print('Best classifier score')
            print(metrics.classification_report(y_test, prediciton))
            print()
            print('Confussion Matrix')
            print(metrics.confusion_matrix(y_test, prediciton))
            print()
            print('f1 score')
            print(metrics.f1_score(y_test, prediciton))
        return classifier


    def process_folder(self, path: str)-> List[List[float]]:
        """
            Retorna o array de features para todas imagens em um diretório

            Args:
            path (str): path da pasta.

            Returns:
            lista de lista de float contendo os feature vector da image
        """
        treinamento = []
        for root, _, files in os.walk(path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                img_feature = self.process_image_file(file_path)
                if img_feature:
                    treinamento.append(img_feature)
        return treinamento
    
    def process_image_file(self, image_path:str) -> List[float]:
        """
            Dado um path de uma imagem, retornar seu feature vector.

            Args:
            image_path (str): path da imagem a ser processada.

            Returns:
            lista (float): vetor de features ou None.
        """
        try:
            image = Image.open(image_path)
            return self.process_image(image)
        except IOError:
            return None
    
    def process_image(self, image: object, blocks=4) -> List[float]:
        """
            Dado um objeto de imagem PIL, retorna seu feture vector.

            Args:
            image (PIL.Image): imagem a ser processada.
            blocks (int, optional): número de blocos pra ser dividido com o espaço RGB.

            Returns:
            lista (float): se sucesso retornar vetor de features, None senão
        """
        if not image.mode == 'RGB':
            return None
        feature = [0] * blocks * blocks * blocks
        pixel_count = 0
        for pixel in image.getdata():
            ridx = int(pixel[0]/(256/blocks))
            gidx = int(pixel[1]/(256/blocks))
            bidx = int(pixel[2]/(256/blocks))
            idx = ridx + gidx * blocks + bidx * blocks * blocks
            feature[idx] += 1
            pixel_count += 1
        return [x/pixel_count for x in feature]

    def save_trained_data(self, svm: object):
        return joblib.dump(svm, 'trainedData.pkl') 
    
    def load_trainded_data(self, path: str):
        return joblib.load(path)

if __name__ == '__main__':
    
    start = time.time()
    classifier = SVM()
    classifier.train_data('/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/LaranjaTangerina/Segment_Images/jseg/colorida/sem_linha/',
    '/home/jonatha/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/LaranjaInfectada/Segment_Images/jseg/colorida/')
    end = time.time()
    classifier.save_trained_data(classifier)
    print("tempo de treinamento: ", end-start)
    # TODO:
    #   - Testar o modelo treinado.
    #   - gerar gráficos.
    #   - escrever o relatório. (enxer linguiça)
    #   - refatorar esta classe e o remover-linha-branca dos algoritmos de segmentacao.