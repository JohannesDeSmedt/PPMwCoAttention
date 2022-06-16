import numpy as np
import pandas as pd
import warnings

from os.path import exists

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

from tensorflow.keras.optimizers import Nadam
from sklearn.metrics import accuracy_score, precision_score
from tensorflow.keras.callbacks import EarlyStopping

from time import perf_counter

from create_inputs import create_case_log_prefix_traces_new
from create_lstms import prefix_plain_lstm, dual_lstm_concat, dual_lstm_mul, dual_lstm_MLB, dual_lstm_MFB

from sklearn.model_selection import KFold


warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 6)
pd.set_option("display.max_rows", 6)
np.random.seed(2)


def fit_and_score(input_train, input_test, y_train, y_test):

    start_time = perf_counter()

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

    if model_type == 'plain_lstm_case':
        model = prefix_plain_lstm(no_act=no_act, lstm_dim=lstm_dim,
                                  length_input=prefix_length)
    if model_type == 'plain_lstm_log':
        model = prefix_plain_lstm(no_act=no_act, lstm_dim=lstm_dim,
                                  length_input=prefix_model)

    if model_type == 'dual_lstm_MLB':
        model = dual_lstm_MLB(no_act=no_act, attention_dim=attention_dim,
                                                  lstm_dim=lstm_dim, length_input=prefix_length,
                                                  prefix_model=prefix_model)
    if model_type == 'dual_lstm_MFB':
        model = dual_lstm_MFB(no_act=no_act, lstm_dim=lstm_dim, length_input=prefix_length, k=k, g=g,
                                                  prefix_model=prefix_model, attention_dim=attention_dim)
    if model_type == 'dual_lstm_concat':
        model = dual_lstm_concat(no_act=no_act, lstm_dim=lstm_dim, length_input=prefix_length,
                                                  prefix_model=prefix_model)
    if model_type == 'dual_lstm_mul':
        model = dual_lstm_mul(no_act=no_act, lstm_dim=lstm_dim, length_input=prefix_length,
                                 prefix_model=prefix_model)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # print(model.summary())

    if dataset == 'italian' or dataset == 'sepsis':
        h = model.fit(input_train, y_train, callbacks=[], validation_split=0.2, epochs=no_total_epochs, verbose=1)
    else:
        h = model.fit(input_train, y_train, callbacks=[early_stopping], validation_split=0.2, epochs=no_total_epochs, verbose=1)

    end_time = perf_counter()

    predictions = model.predict(input_test)
    # convert probabilities/one-hot encoded labels to class prediction
    y_class = y_test.argmax(axis=-1)
    y_pred = predictions.argmax(axis=-1)

    accuracy = accuracy_score(y_class, y_pred)
    precision = precision_score(y_class, y_pred, average='macro')

    print('Fold:', fold)
    print("Accuracy test set: %.2f%%" % (accuracy * 100))
    print('Precision: %.2f%%' % (precision * 100))

    n_epochs = len(h.history['loss'])

    result_line = f'{min_length},{prefix_length},{prefix_model},{model_type},{fold},{accuracy},{precision},{model.count_params()},' \
                  f'{end_time - start_time},{n_epochs},{lstm_dim},{attention_dim},{k},{g}\n'
    result_file = open(result_file_name, 'a')
    result_file.write(result_line)
    result_file.close()

    return accuracy


# should mirror the name of a xes file, e.g., 'bpi12.csv'
dataset = 'bpi12'
result_file_name = f'result_file_kf_ml2_kg_test_{dataset}.csv'

if not exists(result_file_name):
    result_file = open(result_file_name, 'w')
    result_file.write('min_length,prefix_length,prefix_model,model_type,fold,acc,prec,n_params,time,n_epochs,lstm_dim,att_dim,k,g\n')
    result_file.close()

event_log_file = '../datasets/' + dataset + '.xes'
variant = xes_importer.Variants.ITERPARSE
paras = {variant.value.Parameters.MAX_TRACES: 400000}
log = xes_importer.apply(event_log_file, parameters=paras)
no_act = len(pm4py.get_event_attribute_values(log, 'concept:name'))

# minimum prefix length
min_length = 2

# number of epochs
no_total_epochs = 50
# number of folds
fold = 5

kfold = KFold(n_splits=5, shuffle=True)

# first, the full dataset is split
X_train_t_1, X_test_t, X_train_p_1, X_test_p, y_train_ohec_1, y_test_ohec, _ = \
    create_case_log_prefix_traces_new(log, no_activities=no_act, min_length=min_length,
                                      prefix_length=5, prefix_model=5, shuffle=True, random_state=42)

# training sets indices are created to be reused across the models
trains = []

labels = np.argmax(y_train_ohec_1, axis=1)

for train, test in kfold.split(X_train_p_1, labels):
    trains.append((train, test))

# LSTM for log prefix
# t_l => prefix_model
for prefix_model in [2, 5, 10, 20, 50]:

    prefix_length = prefix_model
    # create the training data with the same random state so the same events are considered
    # but with different t_l
    X_train_t, X_test_t, X_train_p, X_test_p, y_train_ohec, y_test_ohec, _ = \
        create_case_log_prefix_traces_new(log, no_activities=no_act, min_length=min_length,
                                          prefix_length=prefix_length, prefix_model=prefix_model, shuffle=True, random_state=42)

    models = ['plain_lstm_log']
    for model in models:

        best_accuracy = 0
        best_lstm_dim = 0
        model_type = model
        for lstm_dim in [8, 16, 32]:
            print('\n\nModel:', model)

            accuracies = []
            fold = 0
            for train, test in trains:
                outcome = fit_and_score([X_train_p[train]], [X_train_p[test]], y_train_ohec[train], y_train_ohec[test])
                fold += 1
                accuracies.append(outcome)

            # store hyperparameters if best model so far
            if outcome > np.mean(accuracies):
                best_accuracy = np.mean(accuracies)
                best_lstm_dim = lstm_dim

        print('Testing...')
        lstm_dim = best_lstm_dim
        outcome = fit_and_score([X_train_p], [X_test_p], y_train_ohec, y_test_ohec)

# LSTM for case-based prefix
# prefix_length => t_t
for prefix_length in [2, 5, 10]:

    prefix_model = prefix_length

    # similar to the log prefix LSTM but with different t_t
    X_train_t, X_test_t, X_train_p, X_test_p, y_train_ohec, y_test_ohec, _ = \
        create_case_log_prefix_traces_new(log, no_activities=no_act, min_length=min_length,
                                          prefix_length=prefix_length, prefix_model=prefix_model, shuffle=True, random_state=42)

    models = ['plain_lstm_case']
    for model in models:

        best_accuracy = 0
        best_lstm_dim = 0
        model_type = model

        print('\n\nModel:', model)
        for lstm_dim in [8, 16, 32]:
            print('LSTM dim:', lstm_dim)

            accuracies = []
            fold = 0
            for train, test in trains:
                outcome = fit_and_score([X_train_t[train]], [X_train_t[test]], y_train_ohec[train], y_train_ohec[test])
                fold += 1
                accuracies.append(outcome)

            if best_accuracy < np.mean(accuracies):
                best_accuracy = np.mean(accuracies)
                best_lstm_dim = lstm_dim

        print('Testing...')
        lstm_dim = best_lstm_dim
        outcome = fit_and_score([X_train_t], [X_test_t], y_train_ohec, y_test_ohec)

# Dual LSTM with concatenation/multiplication/MLB/MFB
for prefix_length in [2, 5, 10]:
    for prefix_model in [prefix_length, int(prefix_length * 2), 50]:
        # see evaluatoin setup in paper
        if prefix_model == 50 and prefix_length != 10:
            continue

        X_train_t, X_test_t, X_train_p, X_test_p, y_train_ohec, y_test_ohec, _ = \
            create_case_log_prefix_traces_new(log, no_activities=no_act, min_length=min_length,
                                              prefix_length=prefix_length, prefix_model=prefix_model, shuffle=True, random_state=42)

        models = ['dual_lstm_mul', 'dual_lstm_concat']
        for model in models:

            best_accuracy = 0
            best_lstm_dim = 0
            model_type = model

            print('\n\nModel:', model)
            for lstm_dim in [8, 16, 32]:
                print('LSTM dim:', lstm_dim)

                accuracies = []
                fold = 0
                for train, test in trains:
                    outcome = fit_and_score([X_train_t[train], X_train_p[train]],
                                            [X_train_t[test], X_train_p[test]], y_train_ohec[train], y_train_ohec[test])
                    fold += 1
                    accuracies.append(outcome)

                if best_accuracy < np.mean(accuracies):
                    best_accuracy = np.mean(accuracies)
                    best_lstm_dim = lstm_dim

            print('Testing...')
            lstm_dim = best_lstm_dim
            outcome = fit_and_score([X_train_t, X_train_p], [X_test_t, X_test_p], y_train_ohec, y_test_ohec)

        # similar to previous models but now with additional attention/factorization hyperparameter
        models = ['dual_lstm_MFB', 'dual_lstm_MLB']
        for model in models:

            best_accuracy = 0
            best_lstm_dim = 0
            best_att_dim = 0
            model_type = model

            print('\n\nModel:', model)
            lstm_dim = 16
            attention_dim = 8

            for lstm_dim in [8, 16, 32]:
                for attention_dim in [3, 5, 7, 9]:
                    if attention_dim > lstm_dim:
                        continue

                    accuracies = []
                    fold = 0
                    for train, test in trains:
                        outcome = fit_and_score([X_train_t[train], X_train_p[train]],
                                                [X_train_t[test], X_train_p[test]], y_train_ohec[train],
                                                y_train_ohec[test])
                        fold += 1
                        accuracies.append(outcome)

                    if best_accuracy < np.mean(accuracies):
                        best_accuracy = np.mean(accuracies)
                        best_lstm_dim = lstm_dim
                        best_att_dim = attention_dim

            print('Testing...')
            lstm_dim = best_lstm_dim
            attention_dim = best_att_dim
            # previously established through hyperparameter search
            # the models are quite insensitive to this number
            k = 5
            g = 4
            outcome = fit_and_score([X_train_t, X_train_p], [X_test_t, X_test_p], y_train_ohec, y_test_ohec)


