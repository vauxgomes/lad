import numpy as np

class LazyPatterns():

    def __init__(self, binarizer, selector):
        self.__Xbin = None
        self.__y = None
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

        return np.array(predictions)

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
        print(f'LazyPatterns Set of Rules [None]:')