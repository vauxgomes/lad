import numpy as np


class MaxPatterns():

    def __init__(self, purity=0.95):
        self.__min_purity = purity
        self.__rules = []

    def get_rules(self):
        return self.__rules

    def __purity(self, y):
        unique, counts = np.unique(y, return_counts=True)
        argmax = np.argmax(counts)

        purity = counts[argmax]/len(y)
        label = unique[argmax]

        return len(y), counts[argmax], purity, label

    def predict(self, X):
        weights = {}

        for r in self.__rules:

            label = r['label']
            weight = r['weight']

            indexes = np.arange(X.shape[0])

            for i, condition in enumerate(r['conditions']):
                att = r['attributes'][i]
                val = r['values'][i]

                if (condition):
                    #print(f'att{att} <= {val}', end=', ')
                    indexes = indexes[np.where(X.T[att, indexes] <= val)]
                else:
                    #print(f'att{att} > {val}', end=', ')
                    indexes = indexes[np.where(X.T[att, indexes] > val)]

            # print(r['label'])

            for i in indexes:
                weights[i] = weights.get(i, {})
                weights[i][label] = weights[i].get(label, 0) + weight

        predictions = []
        for i in range(X.shape[0]):
            if i not in weights:
                predictions.append(2)
            else:
                predictions.append(max(weights[i], key=weights[i].get))

        return np.array(predictions)

    def fit(self, Xbin, y):
        self.__rules.clear()

        rules_weights = []
        labels_weights = {}

        for instance in np.unique(Xbin, axis=0):
            attributes = list(np.arange(instance.shape[0]))

            covered = np.where(
                (Xbin[:, attributes] == instance[attributes]).all(axis=1))
            repet, count, purity, label = self.__purity(y[covered])

            # Choosing rule's attributes
            while len(attributes) > 1 and len(covered) <= Xbin.shape[0]:
                best = None  # Actually, the worst
                __attributes = attributes.copy()

                # Find the best attribute to be removed
                for att in attributes:
                    # Candidate
                    __attributes.remove(att)

                    # Candidate's coverage
                    __covered = np.where(
                        (Xbin[:, __attributes] == instance[__attributes]).all(axis=1))

                    # Stats
                    _, __count, __purity, _ = self.__purity(y[__covered])

                    # Testing candidate
                    if __purity >= self.__min_purity:
                        if __purity > purity or (__purity == purity and __count > count):
                            best = att

                    #
                    __attributes.append(att)

                if best is None:
                    break

                # Update rule
                attributes.remove(best)

                covered = np.where(
                    (Xbin[:, attributes] == instance[attributes]).all(axis=1))
                _, count, purity, label = self.__purity(y[covered])

            # Forming rule object
            r = {
                'label': label,
                'attributes': attributes.copy(),
                'conditions': list(instance[attributes]),
                'purity': purity
            }

            # Storing rule
            if r not in self.__rules:
                self.__rules.append(r)
                rules_weights.append(repet)
            else:
                # When the same rule as build more than once
                rules_weights[self.__rules.index(r)] += repet

            labels_weights[label] = labels_weights.get(label, 0) + repet

        # Reweighting
        for i, r in enumerate(self.__rules):
            r['weight'] = rules_weights[i]/labels_weights[r['label']]

    def adjust(self, binarizer, selector):
        cutpoints = binarizer.get_cutpoints()
        selected = selector.get_selected()

        for r in self.__rules:
            __cutpoints = [cutpoints[i] for i in selected[r['attributes']]]

            r['attributes'].clear()
            r['values'] = []

            for i, c in enumerate(__cutpoints):
                r['attributes'].append(c[0])
                r['values'].append(c[1])


class LazyMaxPatterns():

    def __init__(self, purity=0.95):
        self.__min_purity = purity

        self.__Xbin = None
        self.__y = None
        self.__binarizer = None
        self.__selector = None

    def __purity(self, y):
        if len(y) == 0:
            return 0, 0, 0, None

        unique, counts = np.unique(y, return_counts=True)
        argmax = np.argmax(counts)

        purity = counts[argmax]/len(y)
        label = unique[argmax]

        return len(y), counts[argmax], purity, label

    def predict(self, X):
        Xbin = self.__selector.transform(self.__binarizer.transform(X))
        predictions = []

        for instance in Xbin:
            attributes = list(np.arange(instance.shape[0]))

            covered = np.where(
                (self.__Xbin[:, attributes] == instance[attributes]).all(axis=1))
            repet, count, purity, label = self.__purity(self.__y[covered])

            # Choosing rule's attributes
            while len(attributes) > 1 and len(covered) <= self.__Xbin.shape[0]:
                best = None  # Actually, the worst
                __attributes = attributes.copy()

                # Find the best attribute to be removed
                for att in attributes:
                    # Candidate
                    __attributes.remove(att)

                    # Candidate's coverage
                    __covered = np.where(
                        (self.__Xbin[:, __attributes] == instance[__attributes]).all(axis=1))

                    # Stats
                    _, __count, __purity, _ = self.__purity(
                        self.__y[__covered])

                    # Testing candidate
                    if __purity >= self.__min_purity or purity == 0:
                        if __purity > purity or (__purity == purity and __count > count):
                            best = att

                    #
                    __attributes.append(att)

                if best is None:
                    break

                # Update rule
                attributes.remove(best)

                covered = np.where(
                    (self.__Xbin[:, attributes] == instance[attributes]).all(axis=1))
                count, _, purity, label = self.__purity(self.__y[covered])

            # Get most frequent if no rule was formed
            if purity == 0:
                _, _, _, label = self.__purity(self.__y)

            predictions.append(label)

        return np.array(predictions)

    def fit(self, Xbin, y):
        self.__Xbin = Xbin
        self.__y = y
        # self.__labels = np.unique(y)

    def adjust(self, binarizer, selector):
        self.__binarizer = binarizer
        self.__selector = selector
