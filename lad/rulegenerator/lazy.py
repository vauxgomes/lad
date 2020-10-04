import numpy as np


class LazyPatterns():

    def __init__(self, purity=0.95):
        self.__min_purity = purity

        self.__Xbin = None
        self.__y = None
        self.__binarizer = None
        self.__selector = None

    def predict(self, X):
        Xbin = self.__selector.transform(self.__binarizer.transform(X))
        predictions = []

        for instance in Xbin:
            attributes = []
            literals = list(np.arange(instance.shape[0]))

            # Stats
            _, count, purity, label, discrepancy = self.__get_stats(
                self.__Xbin, self.__y, instance, attributes)

            # Choosing rule's attributes
            while len(literals) > 0:
                best = None
                __attributes = attributes.copy()

                # Find the best attribute to be removed
                for att in literals:
                    # Candidate
                    __attributes.append(att)

                    # Stats
                    _, _, __purity, _, __discrepancy = self.__get_stats(
                        self.__Xbin, self.__y, instance, __attributes)

                    # Testing candidate
                    if __purity > purity or (__purity == purity and __discrepancy < discrepancy):
                        best = att
                        purity = __purity
                        discrepancy = __discrepancy

                    #
                    __attributes.remove(att)

                if best is None:
                    break

                # Update rule
                literals.remove(best)
                attributes.append(best)

                # Stats
                _, count, purity, label, discrepancy = self.__get_stats(
                    self.__Xbin, self.__y, instance, attributes)

            # Get most frequent if no rule was formed
            if purity == 0:
                _, _, _, label, _ = self.__get_stats(
                    self.__Xbin, self.__y, instance, [])

            self.__tmp += 1

            predictions.append(label)

        return np.array(predictions)

    def fit(self, Xbin, y):
        self.__Xbin = Xbin
        self.__y = y
        self.__tmp = 0
        # self.__labels = np.unique(y)

    def adjust(self, binarizer, selector):
        self.__binarizer = binarizer
        self.__selector = selector

    def __get_stats(self, Xbin, y, instance, attributes):
        covered = np.where(
            (Xbin[:, attributes] == instance[attributes]).all(axis=1))
        uncovered = np.setdiff1d(np.arange(Xbin.shape[0]), covered[0])

        if len(covered[0]) == 0:
            counts = 0
            purity = 0
            label = None
        else:
            unique, counts = np.unique(y[covered], return_counts=True)
            argmax = np.argmax(counts)
            purity = counts[argmax]/len(covered[0])
            label = unique[argmax]

            counts = counts[argmax]

        uncovered_class = uncovered[y[uncovered] == label]
        uncovered_other = uncovered[y[uncovered] != label]

        distance_class = np.sum(np.bitwise_xor(
            Xbin[uncovered_class][:, attributes],
            instance[attributes]
        ))

        distance_other = np.sum(np.bitwise_xor(
            Xbin[uncovered_other][:, attributes],
            instance[attributes]
        ))

        discrepancy = (max(1.0, distance_class)/max(1.0, len(uncovered_class)) /
                       max(1.0, distance_other)/max(1.0, len(uncovered_other)))

        return len(covered), counts, purity, label, discrepancy
