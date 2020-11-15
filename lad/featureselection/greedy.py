import numpy as np


class UnWeightedSetCoveringProblem():

    ''' Set covering problem builder '''

    def __init__(self):
        self.__scp = []

    def fit(self, Xbin, y):
        self.__scp.clear()
        labels = np.unique(y)

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):

                # Crossover
                for u in Xbin[y == labels[i]]:
                    for v in Xbin[y == labels[j]]:
                        self.__scp.append(np.bitwise_xor(u, v))

        self.__scp = np.array(self.__scp)

        return self.__scp


class GreedySetCover():
    
    ''' Set covering problem solver '''
    
    def __init__(self):
        self.__selected = []
        self.__scp = None
        
    def get_selected(self):
        return np.array(self.__selected)
        
    def fit(self, Xbin, y):        
        self.__selected.clear()
        
        builder = UnWeightedSetCoveringProblem()
        scp = builder.fit(Xbin, y)
        
        # print('SCP', scp.shape)
                      
        while len(scp):
            sum_ = scp.sum(axis=0)
            att = np.argmax(sum_)
            
            if sum_[att] == 0: break

            scp = np.delete(scp, np.where(scp[:, att]), axis=0)
            self.__selected.append(att)
            
        self.__selected.sort()
            
    def transform(self, Xbin):
        return Xbin[:, self.__selected]
    
    def fit_transform(self, Xbin, y):
        self.fit(Xbin, y)
        return self.transform(Xbin)