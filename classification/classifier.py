# -*- coding: utf-8 -*-
from PIL import Image
from io import StringIO
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import metrics
from sklearn.externals import joblib
from typing import List
import sys
import os
import time


"""
    Classificador SVM (kernel linear)
"""
class SVM:
    """
        Classe representando SVM
    """    
    # def __init__(self, kernel='linear', degree=3, gamma='auto', coef0=0.0,
    #              tol=1e-3, nu=0.5, shrinking=True, cache_size=200,
    #              verbose=False, max_iter=-1, random_state=None):
    #     super(svm.SVC(), self).__init__(
    #         'one_class', kernel, degree, gamma, coef0, tol, 0., nu, 0.,
    #         shrinking, False, cache_size, None, verbose, max_iter,
    #         random_state)

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
        # divido os dados de treinamento em um set de teste e treinamento
        # o set de teste vai conter 30% do total
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(data,
                target, test_size=0.30)
        # definindo uma busca para os melhores parametros 
        parameters = {'kernel': ['linear'], 'C': [1, 10, 100, 1000],
                'gamma': [0.01, 0.001, 0.0001]}
        # procura os melhores parametros para kernel linear variando o C e a gamma
        clf = grid_search.GridSearchCV(svm.SVC(), parameters).fit(x_train, y_train)
        classifier = clf.best_estimator_
        if print_metrics:
            print()
            print('Parametros:', clf.best_params_)
            print()
            print('Score do melhor parametro')
            print(metrics.classification_report(y_test,
                classifier.predict(x_test)))
        return classifier

    def predit_data(self, path_test:str):
        testdata = self.process_folder(path_test)
        return self.predict(testdata)

        
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