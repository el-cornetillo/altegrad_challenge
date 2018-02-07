## A class for the selfTraing knn semi supervised method (implemented in an other class, should be adaptated
## for the altegrad challenge.

class SelfTrainingKnn:
    def __init__(self, n_bunch = 10):
        ':param n_bunch: Number of most confident probabilities to keep'
        self.n_bunch = n_bunch
        pass
    
    def fit(self, X_lab, y_lab, X_unlab):
        X_lab_ = X_lab.copy().tolist()
        y_lab_ = list(y_lab.copy())
        X_unlab_ = X_unlab.copy().tolist()

        self.clf = KNeighborsClassifier()
        self.clf.fit(X_lab_, y_lab_)
        new_predictions = self.clf.predict(np.concatenate([X_lab_, X_unlab_], axis=0))
        old_predictions = np.zeros_like(new_predictions)

        while not np.array_equal(old_predictions, new_predictions):
            old_predictions = new_predictions
            most_confident = np.unravel_index(self.clf.predict_proba(X_unlab_).ravel().argsort(
            )[-min(self.n_bunch, len(X_unlab_)):][::-1], (len(X_unlab_), 2))

            for e in np.array(most_confident).T:
                ## Adds new labelized points to X_lab and y_lab
                X_lab_.append(X_unlab_[e[0]])
                y_lab_.append(e[1])

            # Removes elements from X_unlab. Needs a counter to keep track of the changing size of X_unlab
            n = 0
            for i in np.sort(np.array(most_confident)[0]):
                del X_unlab_[i - n]
                n += 1

            self.clf.fit(X_lab_, y_lab_)

            if len(X_unlab_) > 0:
                new_predictions = self.clf.predict(
                    np.concatenate([X_lab_, X_unlab_], axis=0))
            else:
                new_predictions = old_predictions

        pass

    def predict(self, X):
        return self.clf.predict(X)