import sqlite3
import time

import pandas as pd
import numpy as np
from sklearn import model_selection, svm

#--- VARIABLES ---#
TICKER = "'AAPL'"           # Company ticker of data used in process
TEST_SIZE = 0.2             # Ratio that be test-train data seperated
DATA_FRAC = 1.0             # Ratio that determines used data in process

KERNEL = 'rbf'              # 'linear','poly','rbf','sigmoid','precomputed'

C_GAMMA_TEST = True        # C and Gamma Training or Threshold training
THRESHOLD = 0.5             # Company change threshold in %
C_VALUES = [1, 3, 7]           # C values for SVM
GAMMA_VALUES = [1e-7, 0.1]       # Gamma values for SVM

C_VALUE = 1                 # Works if C_GAMMA_TEST False
GAMMA_VALUE = 0.1          # Works if C_GAMMA_TEST False
# np.linspace(0.1, 2.0, 10)   # Works if C_GAMMA_TEST False
THRESHOLDS = [0.4, 0.5, 0.7]

WRITE_FILE = True          # Write results to file
RESULT_FILE_NAME = time.strftime("svm_collective_test_%y-%m-%d_%H-%M-%S.csv",
                                 time.localtime(time.time()))

#--- DATA PREPARING STARTS ---#
tweet_sql = f"""
    SELECT *
    FROM Tweet_WeightedVector_Collective_nonFiltered
    WHERE ticker_symbol={TICKER}
"""
changes_sql = f"""
    SELECT 
        (CompanyValues_CloseChanges.day_date) as change_date, change_prev_1
    FROM CompanyValues_CloseChanges
    INNER JOIN CompanyValues
        ON change_date=CompanyValues.day_date
            AND CompanyValues_CloseChanges.ticker_symbol=CompanyValues.ticker_symbol
    WHERE CompanyValues_CloseChanges.ticker_symbol IN ({TICKER})
"""

conn = sqlite3.connect('tweets_2015-2019_collective.db')
c = conn.cursor()
c.execute(tweet_sql)
raw_tweet_data = c.fetchall()
c.execute(changes_sql)
changes = dict(c.fetchall())
c.close()


def get_data(threshold):
    data_d = {}
    pos, neg, net = 0, 0, 0
    for raw_tw in raw_tweet_data:
        cur_date = raw_tw[0]
        if not cur_date in data_d:
            data_row = []
            try:
                if changes[cur_date] > threshold:
                    data_row.append(1)
                    pos += 1
                elif changes[cur_date] < -threshold:
                    data_row.append(-1)
                    neg += 1
                else:
                    data_row.append(0)
                    net += 1
            except KeyError:
                continue

            data_row.extend(raw_tw[2:])
            data_d[cur_date] = np.array(data_row, dtype=np.float32)
        else:
            data_d[cur_date][1:] += np.array(raw_tw[2:], dtype=np.float32)
    print(f"pos: {pos}, neg: {neg}, neut: {net}")
    res_data = []
    for k, v in data_d.items():
        v_list = list(v)
        v_list.insert(0, k)
        res_data.append(v_list)

    return res_data


#--- DATA PREPARING ENDS ---#


#--- PREPROCESSING STARTS ---#
column_names = ['post_day', 'change',
                'positive', 'negative', 'uncertainity', 'litigious',
                'constraining', 'superfluous', 'interesting', 'modal']

X_column_names = ['positive', 'negative', 'uncertainity', 'litigious',
                  'constraining', 'superfluous', 'interesting', 'modal']

Y_column_name = 'change'

if C_GAMMA_TEST:
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
        result_file.write(f"Kernel: {KERNEL}\n")
        result_file.write(f"Data ratio: {DATA_FRAC}\n")
        result_file.write(f"Test ratio: {TEST_SIZE}\n")
        result_file.write(
            f"Sample size (test-train): {len(test_X)}-{len(train_X)}\n")
        result_file.write(f"Threshold: {THRESHOLD}\n")
        result_file.write(f"\n#,C_value,gamma_value,accuracy,finish_time(s)\n")

    #--- C and GAMMA TRAINING STARTS ---#
    i = 0
    for c in C_VALUES:
        for gamma in GAMMA_VALUES:
            print(f"{i}:\tC={c:2d}\tγ={gamma:5.4f}\tstarting...")
            clf = svm.SVC(kernel=KERNEL, C=c, gamma=gamma)
            t0 = time.time()
            clf.fit(train_X, train_y)           # training
            dT = time.time() - t0
            test_results = clf.predict(test_X)  # test
            acc = float(np.count_nonzero(test_results == test_y)) / len(test_y)

            if WRITE_FILE:
                result_file.write(f"{i},{c},{gamma},{acc:.5f},{dT:.2f}\n")
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
        result_file.write(f"Kernel: {KERNEL}\n")
        result_file.write(f"Data ratio: {DATA_FRAC}\n")
        result_file.write(f"Test ratio: {TEST_SIZE}\n")
        result_file.write(f"C_value: {C_VALUE}\n")
        result_file.write(f"Gamma_value: {GAMMA_VALUE}\n")
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

        clf = svm.SVC(kernel=KERNEL, C=C_VALUE, gamma=GAMMA_VALUE)
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
