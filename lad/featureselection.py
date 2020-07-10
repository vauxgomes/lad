import numpy as np

class NonWeightedSetCoveringProblem():
    
    ''' Set covering problem builder '''
    
    def __init__(self):
        self.__scp = []
    
    def fit(self, Xbin, y):
        self.__scp.clear()
        
        labels = y.unique()
        
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                
                # Crossover
                for u in Xbin[y == labels[i]].values:
                    for v in Xbin[y == labels[j]].values:
                        self.__scp.append(np.bitwise_xor(u, v))
                        
        self.__scp = np.array(self.__scp)
        # self.__scp = np.delete(self.__scp, 9, axis=1) # quickfix
        
        return self.__scp

class GreedySetCover():
    
    ''' Set covering problem solver '''
    
    def __init__(self):
        self.__selected = []
        self.__scp = None
        
    def fit(self, Xbin, y):        
        self.__selected.clear()
        
        nwscp = NonWeightedSetCoveringProblem()
        scp = nwscp.fit(Xbin, y)
                      
        while len(scp):
            sum_ = scp.sum(axis=0)
            col = np.argmax(sum_)

            scp = np.delete(scp, np.where(scp[:, col]), axis=0)
            self.__selected.append(col)
            
    def transform(self, Xbin):
        return Xbin.iloc[:, self.__selected]
    
    def fit_transform(self, Xbin, y):
        self.fit(Xbin, y)
        return self.transform(Xbin)