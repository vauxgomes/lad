import pandas as pd

class MaxPatterns():
    
    def __init__(self, min_purity=0.95):
        self.__rules  = []
        self.__min_purity = min_purity
        
    def get_rules(self):
        return self.__rules
        
    def __purity(self, y):
        ''' Returns the best purity and label for a given selection of instances '''
        return max([(sum(y == label)/len(y), label) for label in y.unique()])
    
    def __query(self, instance, attributes=None):
        ''' Builds pandas queries based on an instance and a set of attributes '''
        if attributes is None:
            attributes = instance.index
            
        return ' & '.join([f'`{att}`=={instance[att]}' for att in set(attributes)])
    
    def fit(self, Xbin, y):
        ''' Max Patterns Algorithm '''
        self.__rules.clear()        
        
        rules_weights = []
        labels_weights = {}
        
        for k, instance in Xbin.drop_duplicates().iterrows():
            attributes = list(instance.index)
            
            query = self.__query(instance)
            covered = Xbin.query(query)
            #uncovered = Xbin.query(f'not({query})')
            uncovered = Xbin.drop(covered.index, axis=0)
            
            purity, label = self.__purity(y[covered.index])
            
            # Number of duplicates
            count = len(covered)
            
            # Choosing rule's attributes
            while len(attributes) > 1 and len(uncovered) > 0:
                
                best_att = None
                __attributes = attributes.copy()
                
                # Find the best attribute to be removed
                for att in attributes:
                    # Candidate
                    __attributes.remove(att)
                    
                    # Candidate's coverage
                    __query = self.__query(instance, __attributes)
                    __covered = pd.concat([covered, uncovered.query(__query)]) 
                    #__uncovered = Xbin.query(f'not({__query})')
                    uncovered = Xbin.drop(__covered.index, axis=0)
                    
                    #
                    __purity, __label = self.__purity(y[__covered.index])
                    
                    # Choosing best purity of best coverage
                    if __purity >= self.__min_purity and \
                        (__purity > purity or \
                             (__purity == purity and len(__covered) > len(covered))):
                        best_att = att
                    
                    #
                    __attributes.append(att)
                         
                if best_att is None:
                    break
                    
                # Update rule
                attributes.remove(best_att)
                
                query = self.__query(instance, attributes)
                covered = Xbin.query(query)
                uncovered = Xbin.drop(covered.index, axis=0)
                
                purity, label = self.__purity(y[covered.index])
                
                
            rule = {
                'label': label,
                'attributes': attributes.copy(),
                'values': list(instance[attributes].values),
                'purity': purity,
            }
            
            if rule not in self.__rules:
                self.__rules.append(rule)
                rules_weights.append(count)
            else:
                rules_weights[self.__rules.index(rule)] += count
                
            labels_weights[label] = labels_weights.get(label, 0) + count
                
        # Reweighting
        for i, rule in enumerate(self.__rules):
            rule['weight'] = rules_weights[i]/labels_weights[rule['label']]        