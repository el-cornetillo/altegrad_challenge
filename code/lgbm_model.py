import lightgbm as lgb
import numpy as np

from sklearn.model_selection import StratifiedKFold

def train_lgbm()
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    model_count = 0
    params = {}
    params['objective'] = 'binary'
    params['metric'] = 'binary_logloss'
    params['learning_rate'] = 0.02
    params['max_depth'] = 8
    params['num_threads'] = 8
    #params['lambda_l2'] = 10

    for idx_train, idx_val in skf.split(y_train, y_train):
        print("MODEL:", model_count)
        train_x = x_train.tocsr()[idx_train].tocoo()
        train_y = y_train[idx_train]

        val_x = x_train.tocsr()[idx_val].tocoo()
        val_y = y_train[idx_val]

        print(' ')
        print(train_x.shape)
        print(train_y.shape)
        print(val_x.shape)
        print(val_y.shape)
        print(' ')

        d_train = lgb.Dataset(train_x, label=train_y)
        d_val = lgb.Dataset(val_x, label=val_y)

        lgbm = lgb.train(params, 
                     d_train, 
                     num_boost_round=9999999999, 
                     valid_sets=[d_val], 
                     early_stopping_rounds=300,
                     verbose_eval = 100)

        print(' ')
        print(model_count, "validation loss:", lgbm.best_score['valid_0']['binary_logloss'])
        print(' ')

        preds = lgbm.predict(x_test, num_iteration=lgbm.best_iteration)

        submission = pd.DataFrame({"test_id": df_test["id"].values, "is_duplicate": preds})
        submission.to_csv("lgbm_preds" + str(model_count) + ".csv", index=False)

        model_count += 1