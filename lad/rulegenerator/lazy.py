import numpy as np

class LazyPatterns():

    def __init__(self, binarizer, selector):
        self.__Xbin = None
        self.__y = None
        self.__rules = []

        self.__binarizer = binarizer
        self.__selector = selector

    def predict(self, X):
        Xbin = self.__selector.transform(self.__binarizer.transform(X))
        predictions = []

        for instance in Xbin:
            scores = []

            for l in self.__labels:
                attributes = list(np.arange(instance.shape[0]))

                # Stats
                label, confidence, support, lift = self.__get_stats(
                    instance, attributes, l)

                # Choosing rule's attributes
                while len(attributes) > 1:
                    best = None
                    __attributes = attributes.copy()

                    # Find the best attribute to be removed
                    for att in attributes:
                        # Candidate
                        __attributes.remove(att)

                        # Stats
                        _, __confidence, __support, __lift = self.__get_stats(
                            instance, __attributes, l)

                        # Testing candidate
                        if __confidence > confidence \
                            or (__confidence == confidence and __support > support) \
                            or (__confidence == confidence and __support == support and __lift > lift):
                            best = att
                            confidence = __confidence
                            support = __support
                            lift = __lift

                        #
                        __attributes.append(att)

                    if best is None:
                        break

                    # Update rule
                    attributes.remove(best)

                    # Stats
                    label, confidence, support, lift = self.__get_stats(
                        instance, attributes, l)

                # Saving score
                scores.append((label, confidence, support, lift))

            # Best score
            label = sorted(scores, key=lambda x: (x[1], x[2], x[3]))[-1][0]
            predictions.append(label)

            # Forming rule object
            r = {
                'label': label,
                'attributes': attributes.copy(),
                'conditions': list(instance[attributes]),
                'confidence': confidence,
                'support': support,
                'lift': lift
            }

            # Storing rule
            if r not in self.__rules:
                self.__rules.append(r)

        self.__adjust()        
        
        return np.array(predictions)

    def __adjust(self):
        for r in self.__rules:
            conditions = {}
            cutpoints = [self.__binarizer.get_cutpoints()[i] for i in self.__selector.get_selected()[r['attributes']]]

            for i, (att, value) in enumerate(cutpoints):
                condition = conditions.get(att, {})
                symbol = r['conditions'][i]  # True: <=, False: >

                if symbol: condition[symbol] = min(value, condition.get(symbol, value))
                else: condition[symbol] = max(value, condition.get(symbol, value))

                conditions[att] = condition

            r['attributes'].clear()
            r['conditions'].clear()
            r['values'] = []
            
            for att in conditions:
                for condition in conditions[att]:
                    r['attributes'].append(att)
                    r['conditions'].append(condition == '<=')
                    r['values'].append(conditions[att][condition])

        self.__rules.sort(key=lambda x: x['label'])
    
    def predict_proba(self, X):
        predictions = self.predict(X)
        output = np.zeros((len(X), len(np.unique(self.__y))))

        for i in range(len(X)):
            output[i][predictions[i]] = 1

        return output

    def fit(self, Xbin, y):
        self.__Xbin = Xbin
        self.__y = y

        unique, counts = np.unique(y, return_counts=True) 
        self.__labels = {unique[i]: counts[i] for i in range(len(unique))}

    def __get_stats(self, instance, attributes, label):
        covered = np.where(
            (self.__Xbin[:, attributes] == instance[attributes]).all(axis=1))

        confidence = 0
        support = 0
        lift = 2

        if len(covered[0]) > 0:
            unique, counts = np.unique(self.__y[covered], return_counts=True)
            argmax = np.argmax(counts)

            if label is None:
                label = unique[argmax]

            if label in unique:
                confidence = counts[argmax]/sum(counts)
                support = counts[argmax]/self.__Xbin.shape[0]
                lift = (counts[argmax]/len(covered[0]))/(self.__labels[unique[argmax]]/self.__y.shape[0])

        return label, confidence, support, lift

    def __str__(self):
        s = f'LazyPatterns Set of Rules:\n'
        
        for r in self.__rules:
            label = r['label']
            # weight = r['weight']
            conditions = []

            for i, condition in enumerate(r['conditions']):
                att = r['attributes'][i]
                val = r['values'][i]

                if (condition):
                    conditions.append(f'att{att} <= {val:.4}')
                else:
                    conditions.append(f'att{att} > {val:.4}')

            # Label -> CONDITION_1 AND CONDITION_2 AND CONDITION_n
            s += f'{label} \u2192 {" AND ".join(conditions)}\n'

        return s