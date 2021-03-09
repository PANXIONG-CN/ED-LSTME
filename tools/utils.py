from math import sqrt
from numpy import split
from numpy import array
from sklearn.metrics import mean_squared_error
from numpy import concatenate
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
import numpy
from sklearn.preprocessing import MinMaxScaler
from timeit import default_timer as timer


def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train = train.astype(float)
    test = test.astype(float)
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def split_dataset(data, n_steps_out):
    train, test = data[0:46080, 7:26], data[46080:61440, 7:26]
    scaler, train_scaled, test_scaled = scale(train, test)
    train_scaled = array(split(train_scaled, len(train_scaled) / n_steps_out))
    test_scaled = array(split(test_scaled, len(test_scaled) / n_steps_out))
    return scaler, train_scaled, test_scaled


def model_configs():
    n_input = [1 * 97, 2 * 97, 3 * 97]
    n_steps_out = [8, 12, 16]
    n_nodes = [100, 200, 300, 400]
    n_epochs = [50, 100]
    n_batch = [32, 64, 128]

    configs = list()
    for i in n_input:
        for j in n_steps_out:
            for k in n_nodes:
                for l in n_epochs:
                    for m in n_batch:
                        cfg = [i, j, k, l, m]
                        configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs


def evaluate_forecasts(actual, predicted):
    scores = list()
    for i in range(0, actual.shape[1], 4):
        mse = mean_squared_error(actual[:, i, :], predicted[:, i, :])
        rmse = sqrt(mse)
        scores.append(rmse)
    s = 0
    for x in range(actual.shape[0]):
        for y in range(actual.shape[1]):
            for z in range(actual.shape[2]):
                s += (actual[x, y, z] - predicted[x, y, z]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1] * actual.shape[2]))
    return score, scores


def to_supervised(train, n_steps_in, n_steps_out):
    overlop = n_steps_out
    sequences = train.reshape(
        (train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    for i in range(0, len(sequences), overlop):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, 3:]
        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)


def build_model(train, config):
    n_input, n_steps_out, n_nodes, n_epochs, n_batch = config
    train_x, train_y = to_supervised(train, n_input, n_steps_out)
    verbose, epochs, batch_size = 0, n_epochs, n_batch
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    n_features_y = n_features - 3
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], n_features_y))

    model = Sequential()
    model.add(
        LSTM(
            n_nodes,
            activation='relu',
            input_shape=(
                n_timesteps,
                n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(n_nodes, activation='relu', return_sequences=True))
    model.add(LSTM(n_nodes, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_nodes, activation='relu')))
    model.add(TimeDistributed(Dense(n_features_y)))
    model.compile(loss='mse', optimizer='adam')

    model.fit(
        train_x,
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose)

    return model


def forecast(model, history, n_input):
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    input_x = data[-n_input:, :]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    yhat = model.predict(input_x, verbose=0)
    yhat = yhat[0]
    return yhat


def evaluate_model(dataset, is_first, out_file_csv, cfg):
    start = timer()
    print(is_first)
    n_input, n_steps_out, n_nodes, n_epochs, n_batch = cfg
    scaler, train, test = split_dataset(dataset, n_steps_out)
    model = build_model(train, cfg)
    history = [x for x in train]
    predictions = list()

    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        history.append(test[i, :])

    predictions = array(predictions)
    test = test.reshape((test.shape[0] * test.shape[1], test.shape[2]))
    predictions = predictions.reshape(
        (predictions.shape[0] *
         predictions.shape[1],
         predictions.shape[2]))

    predictions = concatenate((test[:, :3], predictions), axis=1)
    predictions = scaler.inverse_transform(predictions)
    test = scaler.inverse_transform(test)

    numpy.savetxt("output/" + str(cfg) + "predictions.csv", predictions, delimiter=",")
    numpy.savetxt("output/" + str(cfg) + "actuals.csv", test, delimiter=",")

    predictions = array(split(predictions, len(predictions) / n_steps_out))
    test = array(split(test, len(test) / n_steps_out))
    score, scores = evaluate_forecasts(test, predictions)
    run_time = timer() - start

    result_to_save = pd.DataFrame(
        [['3', cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], score, scores, run_time]],
        columns=['layers', 'n_input', 'n_steps_out', 'nodes', 'epochs', ' batch',
                 'RSMEs', 'scores per day', 'runtime']
    )
    result_to_save.to_csv(out_file_csv, index=False, mode='a', header=(is_first == 0))

    is_first = 1

    return cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], score, scores, run_time
