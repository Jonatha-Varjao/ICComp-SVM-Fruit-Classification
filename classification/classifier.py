# -*- coding: utf-8 -*-
from sklearn.svm import OneClassSVM

"""
    Classificador SVM (kernel linear)
"""

class SVM(OneClassSVM):
    """
        Classe representando meu SVM. herdando do sklearn OneClassSVM (libsvm)
    """
    
    def __init__(self, kernel='linear', degree=3, gamma='auto', coef0=0.0,
                 tol=1e-3, nu=0.5, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, random_state=None):
        super(OneClassSVM, self).__init__(
            'one_class', kernel, degree, gamma, coef0, tol, 0., nu, 0.,
            shrinking, False, cache_size, None, verbose, max_iter,
            random_state)


    def train_data(self):
        pass

    def predict_data(self):
        pass

if __name__ == '__main__':
    """
    TODO: 
        Treinar com as imagens inteiras ( só pra teste ), dps passar as áreas segmentadas
    """
    pass