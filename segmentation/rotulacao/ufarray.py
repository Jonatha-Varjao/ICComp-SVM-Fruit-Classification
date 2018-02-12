
from itertools import product

class UFarray:
    def __init__(self):
        # Array das label -> armazena as equivalencias
        self.P = []
        # set do meu novo label, quando for criado outro
        self.label = 0

    def rotulacao_pixel(self, labels,height, width, matrizPixel):
        for y, x in product(range(height), range(width)):
            #
            # Condicoes de vizinhas dos meu pixeis:
            #
            #   -------------
            #   | a | b | c |
            #   -------------
            #   | d | e |   |
            #   -------------
            #   |   |   |   |
            #   -------------
            #
            # Se o meu pixel for 'e'
            # a, b, c, e d sao meus vizinhos de interesse
            # 255 branco, 0 = preto
            # pixeis brancos sao ignorados
            # Pixel branco, ignoro
            if matrizPixel[x, y] == 255:
                pass
            # Se o pixel b for preto :
            # a,c,e sao seus vizinhos, logo fazem parte da mesma regiao
            # e como e é vizinho de d, assumo que b = e = d
            elif y > 0 and matrizPixel[x, y-1] == 0:
                labels[x, y] = labels[(x, y-1)]
            # Se o pixel c for preto :
            #    b é seu vizinho, mas a e d não
            #    logo checo a label de 'a' e 'd'
            elif x+1 < width and y > 0 and matrizPixel[x+1, y-1] == 0:
                c = labels[(x+1, y-1)]
                labels[x, y] = c
                # Se a for petro:
                #    logo a e c estão na mesma regiao
                #    adiciono na união (c,a)
                if x > 0 and matrizPixel[x-1, y-1] == 0:
                    a = labels[(x-1, y-1)]
                    self.union(c, a)
                # Se d for petro:
                #    logo d e c estão na mesma região
                #    adiciona na união (c,d)
                elif x > 0 and matrizPixel[x-1, y] == 0:
                    d = labels[(x-1, y)]
                    self.union(c, d)
            # Se a for preto:
            #    sabemos que c e b sao brancos
            #    d e vizinho de a, logo eles tem a mesma label
            #    lgoo seto a label a em e
            elif x > 0 and y > 0 and matrizPixel[x-1, y-1] == 0:
                labels[x, y] = labels[(x-1, y-1)]
            # Se o d for preto:
            #    logo a,b,c são brancos
            #    logo seto a label d em e
            elif x > 0 and matrizPixel[x-1, y] == 0:
                labels[x, y] = labels[(x-1, y)]
            # toda minha vizinhança é branca
            # logo o pixel atual recebe uma nova label
            else: 
                labels[x, y] = self.criaLabel()
        return self.flatten()

    def criaLabel(self):
        r = self.label
        self.label += 1
        self.P.append(r)
        return r    
    # raiz da Union
    def setRoot(self, i, root):
        while self.P[i] < i:
            j = self.P[i]
            self.P[i] = root
            i = j
        self.P[i] = root
    # Localiza o no raiz que contem o no i
    def findRoot(self, i):
        while self.P[i] < i:
            i = self.P[i]
        return i
    
    # Localiza o no raiz que contem o no i
    # comprimo a arvore
    def find(self, i):
        root = self.findRoot(i)
        self.setRoot(i, root)
        return root    
    
    def union(self, i, j):
        if i != j:
            root = self.findRoot(i)
            rootj = self.findRoot(j)
            if root > rootj: root = rootj
            self.setRoot(j, root)
            self.setRoot(i, root)
    
    def flatten(self):
        for i in range(1, len(self.P)):
            self.P[i] = self.P[self.P[i]]
    
    