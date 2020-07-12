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

    def fit(self, Xbin, y):
        self.__rules.clear()

        rules_weights = []
        labels_weights = {}

        for instance in np.unique(Xbin, axis=0):
            attributes = list(range(len(instance)))

            covered = np.where(
                (Xbin[:, attributes] == instance[attributes]).all(axis=1))
            count, _, purity, label = self.__purity(y[covered])

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
                    __count, _, __purity, _ = self.__purity(y[__covered])

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
                count, _, purity, label = self.__purity(y[covered])

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
                rules_weights.append(count)
            else:
                # When the same rule as build more than once
                rules_weights[self.__rules.index(r)] += count

            labels_weights[label] = labels_weights.get(label, 0) + count

        # Reweighting
        for i, r in enumerate(self.__rules):
            r['weight'] = rules_weights[i]/labels_weights[r['label']]
