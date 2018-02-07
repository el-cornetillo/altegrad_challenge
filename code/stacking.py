import numpy as np
import scipy.stats as st
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss


def stacking(models, X_train, y_train, X_test, method = 'not_proba',
             metric=None, n_folds=10, stratified=True,
             shuffle=False, random_state=0, verbose=0):

    # Specify default metric for cross-validation
    if metric is None:
        metric = accuracy_score
    else:
        metric = metric
        
    # Print metric
    if verbose > 0:
        print('metric: [%s]\n' % metric.__name__)
        
    # Split indices to get folds (stratified can be used only for classification)
    if stratified:
        kf = StratifiedKFold(n_splits = n_folds, shuffle = shuffle, random_state = random_state)
    else:
        kf = KFold(n_splits = n_folds, shuffle = shuffle, random_state = random_state)

    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))
    
    # Loop across models
    for model_counter, model in enumerate(models):
        if verbose > 0:
            print('model %d: [%s]' % (model_counter, model.__class__.__name__))
            
        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        
        # Loop across folds
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            
            # Fit 1-st level model
            model = model.fit(X_tr, y_tr)
            # Predict out-of-fold part of train set
            if method is not 'proba':
                S_train[te_index, model_counter] = model.predict(X_te)
            else:
                S_train[te_index, model_counter] = model.predict_proba(X_te)[:,1]
            # Predict full test set
            if method is not 'proba':
                S_test_temp[:, fold_counter] = model.predict(X_test)
            else:
                S_test_temp[:, fold_counter] = model.predict_proba(X_test)[:,1]
            
            if verbose > 1:
                print('    fold %d: [%.8f]' % (fold_counter, metric(y_te, S_train[te_index, model_counter])))
                
        # Compute mean or mode of predictions for test set
        S_test[:, model_counter] = st.mode(S_test_temp, axis = 1)[0].ravel()
            
        if verbose > 0:
            print('    ----')
            print('    MEAN :   [%.8f]\n' % (metric(y_train, S_train[:, model_counter])))

    return S_train, S_test