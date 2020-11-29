import sqlite3
import time
import datetime

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

#--- VARIABLES ---#
TICKER = "'AAPL'"           # Company ticker of data used in process
TEST_SIZE = 0.2             # Ratio that be test-train data seperated
DATA_FRAC = 1.0             # Ratio that determines used data in process

WRITE_FILE = True          # Write results to file
RESULT_FILE_NAME = time.strftime("random_forest_test_%y-%m-%d_%H-%M-%S.csv",
                                 time.localtime(time.time()))

THRESH_TEST = False

# Threshold testing values - when THRESH_TEST = False
THRESHOLDS = np.linspace(0, 10, 50)
PARAMETER = {'n_estimators': 100, 'max_depth': 7, 'criterion': 'entropy'}

# Parameter testing values - when THRESH_TEST = True
THRESHOLD = 1             # Company change threshold in %
PARAMETERS = [{'n_estimators': 10, 'max_depth': 3, 'criterion': 'gini'},
              {'n_estimators': 10, 'max_depth': 5, 'criterion': 'gini'},
              {'n_estimators': 10, 'max_depth': 7, 'criterion': 'gini'},
              {'n_estimators': 10, 'max_depth': 17, 'criterion': 'gini'},
              {'n_estimators': 10, 'max_depth': 47, 'criterion': 'gini'},
              {'n_estimators': 100, 'max_depth': 3, 'criterion': 'gini'},
              {'n_estimators': 100, 'max_depth': 5, 'criterion': 'gini'},
              {'n_estimators': 100, 'max_depth': 7, 'criterion': 'gini'},
              {'n_estimators': 100, 'max_depth': 17, 'criterion': 'gini'},
              {'n_estimators': 100, 'max_depth': 47, 'criterion': 'gini'},
              {'n_estimators': 1000, 'max_depth': 3, 'criterion': 'gini'},
              {'n_estimators': 1000, 'max_depth': 5, 'criterion': 'gini'},
              {'n_estimators': 1000, 'max_depth': 7, 'criterion': 'gini'},
              {'n_estimators': 1000, 'max_depth': 17, 'criterion': 'gini'},
              {'n_estimators': 1000, 'max_depth': 47, 'criterion': 'gini'},
              {'n_estimators': 10, 'max_depth': 3, 'criterion': 'entropy'},
              {'n_estimators': 10, 'max_depth': 5, 'criterion': 'entropy'},
              {'n_estimators': 10, 'max_depth': 7, 'criterion': 'entropy'},
              {'n_estimators': 10, 'max_depth': 17, 'criterion': 'entropy'},
              {'n_estimators': 10, 'max_depth': 47, 'criterion': 'entropy'},
              {'n_estimators': 100, 'max_depth': 3, 'criterion': 'entropy'},
              {'n_estimators': 100, 'max_depth': 5, 'criterion': 'entropy'},
              {'n_estimators': 100, 'max_depth': 7, 'criterion': 'entropy'},
              {'n_estimators': 100, 'max_depth': 17, 'criterion': 'entropy'},
              {'n_estimators': 100, 'max_depth': 47, 'criterion': 'entropy'},
              {'n_estimators': 1000, 'max_depth': 3, 'criterion': 'entropy'},
              {'n_estimators': 1000, 'max_depth': 5, 'criterion': 'entropy'},
              {'n_estimators': 1000, 'max_depth': 7, 'criterion': 'entropy'},
              {'n_estimators': 1000, 'max_depth': 17, 'criterion': 'entropy'},
              {'n_estimators': 1000, 'max_depth': 47, 'criterion': 'entropy'}]

#--- DATA PREPARING STARTS ---#
tweet_sql = f"""
    SELECT post_date, Tweet_WeightedVector.*
    FROM Tweet_WeightedVector
    INNER JOIN Tweet
        ON Tweet.tweet_id = Tweet_WeightedVector.tweet_id
    WHERE Tweet_WeightedVector.ticker_symbol IN ({TICKER})
"""
changes_sql = f"""
    SELECT 
        (CompanyValues_CloseChanges.day_date) as change_date, change_prev_3
    FROM CompanyValues_CloseChanges
    INNER JOIN CompanyValues
        ON change_date=CompanyValues.day_date
            AND CompanyValues_CloseChanges.ticker_symbol=CompanyValues.ticker_symbol
    WHERE CompanyValues_CloseChanges.ticker_symbol IN ({TICKER})
"""

conn = sqlite3.connect('tweets_2015-2019.db')
c = conn.cursor()
c.execute(tweet_sql)
raw_tweet_data = c.fetchall()
c.execute(changes_sql)
changes = dict(c.fetchall())
c.close()


def get_data(threshold):
    res_data = []
    for raw_tw in raw_tweet_data:
        cur_date = datetime.date.fromtimestamp(raw_tw[0]).__str__()
        data_row = []
        try:
            if changes[cur_date] > threshold:
                data_row.append(1)
            elif changes[cur_date] < -threshold:
                data_row.append(-1)
            else:
                data_row.append(0)
        except KeyError:
            continue

        data_row.append(cur_date)

        data_row.extend(raw_tw[1:])
        res_data.append(data_row)
    return res_data

#--- DATA PREPARING ENDS ---#


#--- PREPROCESSING STARTS ---#
column_names = ['change', 'post_day', 'tweet_id', 'ticker_symbol',
                'positive', 'negative', 'uncertainity', 'litigious',
                'constraining', 'superfluous', 'interesting', 'modal']

X_column_names = ['positive', 'negative', 'uncertainity', 'litigious',
                  'constraining', 'superfluous', 'interesting', 'modal']

Y_column_name = 'change'

if not THRESH_TEST:
    data = get_data(THRESHOLD)
    tweets = pd.DataFrame(data, columns=column_names).sample(frac=DATA_FRAC)

    train_data, test_data = model_selection.train_test_split(
        tweets, test_size=TEST_SIZE,)

    train_X = np.array(train_data[X_column_names])
    train_y = np.array(train_data[Y_column_name])

    test_X = np.array(test_data[X_column_names])
    test_y = np.array(test_data[Y_column_name])

    #--- PREPROCESSING ENDS ---#

    #--- C and GAMMA RESULT FILE PREPARING ---#
    if WRITE_FILE:
        result_file = open(RESULT_FILE_NAME, "w")
        result_file.write(f"Company: {TICKER}\n")
        result_file.write(f"Method: Random Forest\n")
        result_file.write(f"Data ratio: {DATA_FRAC}\n")
        result_file.write(f"Test ratio: {TEST_SIZE}\n")
        result_file.write(
            f"Sample size (test-train): {len(test_X)}-{len(train_X)}\n")
        result_file.write(f"Threshold: {THRESHOLD}\n")
        result_file.write(f"\n#,parameters,accuracy,finish_time(s)\n")

    #--- C and GAMMA TRAINING STARTS ---#
    i = 0
    for n in PARAMETERS:
        print(f"{i}:\t{n}\tstarting...")
        clf = RandomForestClassifier(**n, max_features=1)

        t0 = time.time()
        clf.fit(train_X, train_y)           # training
        dT = time.time() - t0
        test_results = clf.predict(test_X)  # test
        acc = float(np.count_nonzero(test_results == test_y)) / len(test_y)

        if WRITE_FILE:
            result_file.write(f"{i},{n},{acc:.5f},{dT:.2f}\n")
            result_file.flush()
        i += 1

        train_data, test_data = model_selection.train_test_split(
            tweets, test_size=TEST_SIZE)

        train_X = np.array(train_data[X_column_names])
        train_y = np.array(train_data[Y_column_name])

        test_X = np.array(test_data[X_column_names])
        test_y = np.array(test_data[Y_column_name])

        print(f"\taccuracy: {acc:.5f}\tfinished: {dT:.2f}s")

    #--- C and GAMMA TRAINING ENDS ---#

else:
    #--- THRESHOLD RESULT FILE PREPARING ---#
    if WRITE_FILE:
        result_file = open(RESULT_FILE_NAME, "w")
        result_file.write(f"Company: {TICKER}\n")
        result_file.write(f"Method: Random Forest\n")
        result_file.write(f"Data ratio: {DATA_FRAC}\n")
        result_file.write(f"Test ratio: {TEST_SIZE}\n")
        result_file.write(f"parameter: {PARAMETER}\n")
        result_file.write(f"\n#,threshold,accuracy,finish_time(s)\n")

    #--- THRESHOLD TRIALS STARTS ---#
    i = 0
    for thres in THRESHOLDS:
        data = get_data(thres)
        tweets = pd.DataFrame(
            data, columns=column_names).sample(frac=DATA_FRAC)

        train_data, test_data = model_selection.train_test_split(
            tweets, test_size=TEST_SIZE)

        train_X = np.array(train_data[X_column_names])
        train_y = np.array(train_data[Y_column_name])

        test_X = np.array(test_data[X_column_names])
        test_y = np.array(test_data[Y_column_name])
        print(f"{i}:\tThreshold={thres:.3f}\tstarting...")

        clf = RandomForestClassifier(max_features=1, **PARAMETER)
        t0 = time.time()
        clf.fit(train_X, train_y)           # training
        dT = time.time() - t0
        test_results = clf.predict(test_X)  # test
        acc = float(np.count_nonzero(test_results == test_y)) / len(test_y)

        if WRITE_FILE:
            result_file.write(f"{i},{thres},{acc:.5f},{dT:.2f}\n")
            result_file.flush()
        i += 1

        print(f"\taccuracy: {acc:.5f}\tfinished: {dT:.2f}s")
    #--- THRESHOLD TRIALS ENDS ---#
