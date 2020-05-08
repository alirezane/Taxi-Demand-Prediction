import pandas as pd
import datetime
import numpy as np
import os
import pickle
import math
from math import sqrt
from shutil import copyfile
import sklearn.metrics as skmets
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, TimeDistributed, SimpleRNN, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers, regularizers
import sklearn.metrics as skmets
import tensorflow as tf
from keras import backend as K

pd.options.mode.chained_assignment = None
correct_error = 2
correct_percentage = 10
huber_clip_delta = 0.6


def is_holiday(created_at):
    holidays = ["2017-01-10", "2017-03-02", "2017-03-19", "2017-03-20", "2017-03-21", "2017-03-22", "2017-03-23",
                "2017-03-24", "2017-03-25",
                "2017-03-26", "2017-03-27", "2017-03-28", "2017-03-29", "2017-03-30", "2017-03-31", "2017-04-01",
                "2017-04-02", "2017-04-11",
                "2017-04-25", "2017-06-04", "2017-06-05", "2017-06-26", "2017-06-27", "2017-07-20", "2017-09-09",
                "2017-09-30", "2017-10-01",
                "2017-11-09", "2017-11-19", "2017-11-27", "2017-12-06", "2017-12-15"]

    if int((datetime.datetime.weekday(created_at) == 4)):
        return 3
    elif int((datetime.datetime.weekday(created_at) == 3)):
        return 1
    elif str(datetime.datetime.date(created_at)) in holidays:
        return 2
    else:
        return 0


def normalize(x, mean, sd):
    return (x - mean) / sd


def denormalize(x, mean, sd):
    return x * sd + mean


def mape(y_true, y_pred):
    df = pd.DataFrame()
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df = df[df['y_true']!= 0]
    y_true = np.array(df['y_true'])
    y_pred = np.array(df['y_pred'])
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def corrects(actual, predicted, corrects_error=correct_error, corrects_percentage=correct_percentage):
  df = pd.DataFrame()
  df['actual'] = actual
  df['predicted'] = predicted
  df['e'] = abs(df['actual']-df['predicted'])
  correct = df[(df['e']/df['actual']<=(corrects_percentage/100)) | (df['e']<=corrects_error)]
  correct = correct.shape[0]
  total = df.shape[0]
  return correct/total


def calculate_errors(data, target, pred):
    target_labels = [col for col in data if (col.startswith(target) and not col.endswith(')'))]
    pred_labels = [col for col in data if col.startswith(pred)]
    targets = np.array(data[target_labels]).flatten()
    preds = np.array(data[pred_labels]).flatten()
    print("Errors for " + pred + ":\n")
    print("MAE : ", skmets.mean_absolute_error(targets, preds), "\n")
    print("RMSE: ", sqrt(skmets.mean_squared_error(targets, preds)), "\n")
    print("MAPE: ", mape(targets, preds), "\n")
    print("Corrects: ", corrects(targets, preds), "\n")


def calculate_arrays_errors(targets, preds):
    mae = skmets.mean_absolute_error(targets, preds)
    rmse = sqrt(skmets.mean_squared_error(targets, preds))
    MAPE = mape(targets, preds)
    correct = corrects(targets, preds)
    print("MAE : {}   RMSE: {}    MAPE: {}    Corrects:{}".format(mae, rmse, MAPE, correct))


def mse_loss(y_true, y_pred):
    error = y_true - y_pred
    squared_loss = tf.keras.backend.square(error)
    return squared_loss


def mse_loss_mean(y_true, y_pred):
    return tf.keras.backend.mean(mse_loss(y_true, y_pred))


def result_file(data, model_path, target, model_names):
    target_labels = [col for col in data if (col.startswith(target) and not col.endswith(')'))]
    output_text = model_path + "/results.txt"
    f = open(output_text, 'w')
    for model_name in model_names:
        pred_labels = [col for col in data if col.startswith(model_name)]
        targets = np.array(data[target_labels]).flatten()
        preds = np.array(data[pred_labels]).flatten()
        f.write("{}:\n".format(model_name))
        f.write("MAE : {}\n".format(skmets.mean_absolute_error(targets, preds)))
        f.write("RMSE: {}\n".format(sqrt(skmets.mean_squared_error(targets, preds))))
        f.write("MAPE: {}\n".format(mape(targets, preds)))
        f.write("Corrects: {}\n".format(corrects(targets, preds)))
    f.close()


def make_model_dir(model_dir_path, code_file_name):
    directory_created = False
    model_number = 1
    dir_path = model_dir_path
    while directory_created==False:
        model_dir_path = dir_path + "/model-" + str(model_number)
        if os.path.exists(model_dir_path):
            model_number = model_number + 1
        else:
            os.makedirs(model_dir_path)
            directory_created = True
    copyfile("./" + code_file_name + ".py", model_dir_path + "./" + code_file_name + ".py")
    return model_dir_path


class GenerateLSTMDate():
    def __init__(self, data_file_path, train_start_date, train_end_date, val_start_date, val_end_date,
                 interval, seq_len, test_start_date, test_end_date, per_day_limit, normalize=True):
        self.interval = interval
        self.seq_len = seq_len
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.val_start_date = val_start_date
        self.val_end_date = val_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.data_file_path = data_file_path
        self.per_day_limit = per_day_limit
        self.load_dataset()
        self.generate_dataset()
        self.unnormalized_data = self.dataset
        if normalize:
            self.normalize_data()
        self.split_Train_Test()
        self.split_input_output_reshape()

    def load_dataset(self):
        dataset = pd.read_csv(self.data_file_path)
        dataset['createdAt'] = pd.to_datetime(dataset['createdAt'], format='%Y-%m-%d %H:%M:%S')
        days = (dataset.createdAt.max().date() - dataset.createdAt.min().date()).days
        df = dataset.groupby(['region'], as_index=False).agg({'ride_count': {"average": ['sum']}})
        df.columns = df.columns.droplevel(1)
        df.average = df.average / days
        # regions = df[df.average >= self.per_day_limit].region.values
        regions = [16 * a + b for a in range(7, 13) for b in range(7, 13)]
        dataset = dataset[dataset.region.isin(regions)]
        dataset = dataset[['createdAt', 'region', 'ride_count']]
        dataset = dataset[dataset.createdAt >= self.train_start_date]
        self.regions = dataset.region.unique()
        self.mean = dataset.ride_count.mean()
        self.std = dataset.ride_count.std()
        self.dataset = dataset.groupby('region', as_index=False)

    def generate_dataset(self):
        targets = pd.DataFrame()
        targets['createdAt'] = self.dataset.get_group(self.regions[0])['createdAt']
        targets = pd.DataFrame()
        targets['createdAt'] = self.dataset.get_group(self.regions[0])['createdAt']
        targets['is_holiday'] = targets.createdAt.apply(lambda x: is_holiday(x))
        targets['is_holiday'] = (targets['is_holiday'] - targets['is_holiday'].min()) / (
            targets['is_holiday'].max() - targets['is_holiday'].min())
        # Set Day of week
        targets['weekdaynum'] = targets.createdAt.apply(datetime.datetime.weekday)
        targets['weekdaynum'] = (targets['weekdaynum'] - targets['weekdaynum'].min()) / (
            targets['weekdaynum'].max() - targets['weekdaynum'].min())
        # Set time interval number
        targets['timeslot'] = targets.createdAt.apply(lambda x: (x.hour * 2 + x.minute / 30))
        targets['sin_time'] = targets.timeslot.apply(lambda x: math.sin(2 * math.pi * x / (24 * (60 / self.interval))))
        targets['cos_time'] = targets.timeslot.apply(lambda x: math.cos(2 * math.pi * x / (24 * (60 / self.interval))))
        targets = targets.drop(['timeslot'], axis=1)
        for region in self.regions:
            df = self.dataset.get_group(region)[['createdAt', 'ride_count']]
            col_name = 'region_' + str(region)
            df.columns = ['createdAt', col_name]
            targets = pd.merge(targets, df, how='inner', on=['createdAt'])
        request_cols = [col for col in targets.columns if col.startswith('region')]
        for col in request_cols:
            label = col + "_(t+1)"
            targets[label] = targets[col].shift(-1)
        self.dataset = targets

    def normalize_data(self):
        self.dataset.dropna(axis=0, how='any', inplace=True)
        targets_label = [col for col in self.dataset.columns if col.startswith('region')]
        labels = list(targets_label)
        for label in labels:
            self.dataset[label] = self.dataset[label].apply(lambda x: normalize(x, self.mean, self.std))

    def split_Train_Test(self):
        self.test = self.dataset[(self.dataset['createdAt'] >= self.test_start_date) &
                                 (self.dataset['createdAt'] < self.test_end_date)]
        self.train = self.dataset[(self.dataset['createdAt'] >= self.train_start_date) &
                                  (self.dataset['createdAt'] < self.train_end_date)]
        self.val = self.dataset[(self.dataset['createdAt'] >= self.val_start_date) &
                                (self.dataset['createdAt'] < self.val_end_date)]

    def split_input_output_reshape(self):
        targets_label = [col for col in self.dataset.columns[1:] if col.endswith('(t+1)')]
        features_label = [col for col in self.dataset.columns[1:] if not col.endswith('(t+1)')]
        train_X, train_y = self.train[features_label].values, self.train[targets_label].values
        test_X, test_y = self.test[features_label].values, self.test[targets_label].values
        val_X, val_y = self.val[features_label].values, self.val[targets_label].values
        seq_len = self.seq_len

        train_X_res = train_X.shape[0] % seq_len
        train_X = train_X[train_X_res:]

        train_y_res = train_y.shape[0] % seq_len
        train_y = train_y[train_y_res:]

        self.train = self.train[train_X_res:]

        test_X_res = test_X.shape[0] % seq_len
        test_X = test_X[test_X_res:]

        test_y_res = test_y.shape[0] % seq_len
        test_y = test_y[test_y_res:]

        self.test = self.test[test_X_res:]

        val_X_res = val_X.shape[0] % seq_len
        val_X = val_X[val_X_res:]

        val_y_res = val_y.shape[0] % seq_len
        val_y = val_y[val_y_res:]

        self.val = self.val[val_X_res:]

        self.train_X = train_X.reshape((int(train_X.shape[0] / seq_len), seq_len, train_X.shape[1]))
        self.test_X = test_X.reshape((int(test_X.shape[0] / seq_len), seq_len, test_X.shape[1]))
        self.val_X = val_X.reshape((int(val_X.shape[0] / seq_len), seq_len, val_X.shape[1]))
        self.train_y = train_y.reshape((int(train_y.shape[0] / seq_len), seq_len, train_y.shape[1]))
        self.test_y = test_y.reshape((int(test_y.shape[0] / seq_len), seq_len, test_y.shape[1]))
        self.val_y = val_y.reshape((int(val_y.shape[0] / seq_len), seq_len, val_y.shape[1]))


class GenrateOtherMethodsData():
    def __init__(self, data_file_path, lookBack, train_start_date, train_end_date, val_start_date, val_end_date,
                 test_start_date, test_end_date, interval, per_day_limit, normalize=True, input_3d=True):
        self.lookBack = lookBack
        self.interval = interval
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.val_start_date = val_start_date
        self.val_end_date = val_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.data_file_path = data_file_path
        self.per_day_limit = per_day_limit
        self.input_3d = input_3d
        self.load_dataset()
        self.generate_targets()
        self.generate_request_timeseries()
        self.generate_timestamps()
        self.generate_dataset()
        self.unnormalized_data = self.dataset
        if normalize:
            self.normalize_data()
        self.split_Train_Test()
        self.split_input_output_reshape()

    def load_dataset(self):
        dataset = pd.read_csv(self.data_file_path)
        dataset['createdAt'] = pd.to_datetime(dataset['createdAt'], format='%Y-%m-%d %H:%M:%S')
        days = (dataset.createdAt.max().date() - dataset.createdAt.min().date()).days
        df = dataset.groupby(['region'], as_index=False).agg({'ride_count': {"average": ['sum']}})
        df.columns = df.columns.droplevel(1)
        df.average = df.average/days
        # regions = df[df.average >= self.per_day_limit].region.values
        regions = [16 * a + b for a in range(7, 13) for b in range(7, 13)]
        dataset = dataset[dataset.region.isin(regions)]
        dataset = dataset[['createdAt', 'region', 'ride_count']]
        self.regions = dataset.region.unique()
        self.mean = dataset.ride_count.mean()
        self.std = dataset.ride_count.std()
        self.dataset = dataset.groupby('region', as_index=False)

    def generate_targets(self):
        targets = pd.DataFrame()
        targets['createdAt'] = self.dataset.get_group(self.regions[0])['createdAt']
        for region in self.regions:
            df = self.dataset.get_group(region)[['createdAt', 'ride_count']]
            col_name = 'target_' + str(region)
            df.columns = ['createdAt', col_name]
            targets = pd.merge(targets, df, how='inner', on=['createdAt'])
        self.targets = targets.loc[self.lookBack:]

    def generate_request_timeseries(self):
        request_series = pd.DataFrame()
        request_series['createdAt'] = self.dataset.get_group(self.regions[0])['createdAt']
        for region in self.regions:
            df = self.dataset.get_group(region)[['createdAt', 'ride_count']]
            col_name = 'request_' + str(region)
            df.columns = ['createdAt', col_name]
            request_series = pd.merge(request_series, df, how='inner', on=['createdAt'])
        temp_request_series = request_series.copy()
        temp_request_series.columns = ['createdAt'] + list(
            temp_request_series.columns[1:] + '_(t-' + str(self.lookBack) + ')')
        for i in range(1, self.lookBack):
            df2 = request_series[request_series.columns[1:]].shift(-i)
            df2.columns = list(request_series.columns[1:] + '_(t-' + str(self.lookBack - i) + ')')
            temp_request_series = pd.concat([temp_request_series, df2], axis=1)
        request_series = temp_request_series.copy()
        request_series['createdAt'] = request_series.createdAt.shift(-self.lookBack)
        self.request_series = request_series

    def generate_timestamps(self):
        timestamps = pd.DataFrame()
        timestamps['createdAt'] = self.dataset.get_group(self.regions[0])['createdAt']
        timestamps['is_holiday'] = timestamps.createdAt.apply(lambda x: is_holiday(x))
        timestamps['is_holiday'] = (timestamps['is_holiday'] - timestamps['is_holiday'].min()) / (
        timestamps['is_holiday'].max() - timestamps['is_holiday'].min())
        # Set Day of week
        timestamps['weekdaynum'] = timestamps.createdAt.apply(datetime.datetime.weekday)
        timestamps['weekdaynum'] = (timestamps['weekdaynum'] - timestamps['weekdaynum'].min()) / (
        timestamps['weekdaynum'].max() - timestamps['weekdaynum'].min())
        # Set time interval number
        timestamps['timeslot'] = timestamps.createdAt.apply(lambda x: (x.hour * 2 + x.minute / 30))
        timestamps['sin_time'] = timestamps.timeslot.apply(lambda x: math.sin(2 * math.pi * x / (24 * (60 / self.interval))))
        timestamps['cos_time'] = timestamps.timeslot.apply(lambda x: math.cos(2 * math.pi * x / (24 * (60 / self.interval))))
        timestamps = timestamps.drop(['timeslot'], axis=1)
        time_lookback = 1
        temp_timestamps = timestamps.copy()
        temp_timestamps.columns = ['createdAt'] + list(temp_timestamps.columns[1:] + '(t-' + str(time_lookback) + ')')
        for i in range(1, time_lookback):
            df2 = timestamps[timestamps.columns[1:]].shift(-i)
            df2.columns = list(timestamps.columns[1:] + '(t-' + str(time_lookback - i) + ')')
            temp_timestamps = pd.concat([temp_timestamps, df2], axis=1)
        timestamps = temp_timestamps
        timestamps['createdAt'] = timestamps.createdAt.shift(-time_lookback)
        self.timestamps = timestamps

    def generate_dataset(self):
        dataset = self.targets.copy()
        for i in range(self.lookBack, 0, -1):
            label = '(t-' + str(i) + ')'
            time_cols = ['createdAt'] + [col for col in self.timestamps.columns if col.endswith(label)]
            req_cols = ['createdAt'] + [col for col in self.request_series.columns if col.endswith(label)]
            dataset = pd.merge(dataset, self.timestamps[time_cols], how='left', on=['createdAt'])
            dataset = pd.merge(dataset, self.request_series[req_cols], how='left', on=['createdAt'])
        self.dataset = dataset

    def normalize_data(self):
        self.dataset.dropna(axis=0, how='any', inplace=True)
        targets_label = [col for col in self.dataset.columns if col.startswith('target')]
        requests_label = [col for col in self.dataset.columns if col.startswith('request')]
        labels = list(targets_label) + list(requests_label)
        for label in labels:
            self.dataset[label] = self.dataset[label].apply(lambda x: normalize(x, self.mean, self.std))

    def split_Train_Test(self):
        self.test = self.dataset[(self.dataset['createdAt'] >= self.test_start_date) &
                                 (self.dataset['createdAt'] < self.test_end_date)]
        self.val = self.dataset[(self.dataset['createdAt'] >= self.val_start_date) &
                                 (self.dataset['createdAt'] < self.val_end_date)]
        self.train = self.dataset[(self.dataset['createdAt'] >= self.train_start_date) &
                                  (self.dataset['createdAt'] < self.train_end_date)]

    def split_input_output_reshape(self):
        features_label = [col for col in self.dataset.columns[1:] if col.endswith(')')]
        features = self.dataset.columns[(1+len(self.regions)):]
        targets_label = [col for col in self.targets.columns if col.startswith('target')]
        train_X, train_y = self.train[features_label].values, self.train[targets_label].values
        test_X, test_y = self.test[features_label].values, self.test[targets_label].values
        valid_X, valid_y = self.val[features_label].values, self.val[targets_label].values
        if self.input_3d:
            self.train_X = train_X.reshape((train_X.shape[0], self.lookBack, int(train_X.shape[1] / self.lookBack)))
            self.test_X = test_X.reshape((test_X.shape[0], self.lookBack, int(test_X.shape[1] / self.lookBack)))
            self.val_X = valid_X.reshape((valid_X.shape[0], self.lookBack, int(valid_X.shape[1] / self.lookBack)))
        else:
            self.train_X = train_X
            self.test_X = test_X
            self.val_X = valid_X
        self.train_y = train_y.reshape((train_y.shape[0], len(self.regions)))
        self.test_y = test_y.reshape((test_y.shape[0], len(self.regions)))
        self.val_y = valid_y.reshape((valid_y.shape[0], len(self.regions)))


class Lstm():
    def __init__(self, train_x, train_y, valid_x, valid_y, num_neurons,
                 epoch, batch_size, lr, loss, activation, patience, num_regions):
        # Model properties
        self.num_neurons = num_neurons
        self.epoch = epoch
        self.loss = loss
        self.lr = lr
        self.num_regions = num_regions
        self.batch_size = batch_size
        self.act_func = activation
        self.patience = patience
        self.train_X = train_x
        self.train_y = train_y
        self.valid_X = valid_x
        self.valid_y = valid_y
        self.make_model()

    def make_model(self):
        # design network
        model = Sequential()
        if len(self.num_neurons) == 1:
            model.add(LSTM(self.num_neurons[0], input_shape=(None, self.train_X.shape[2]),
                           return_sequences=True, activation=self.act_func))
        else:
            model.add(LSTM(self.num_neurons[0], input_shape=(None, self.train_X.shape[2]),
                           return_sequences=True, activation=self.act_func))
            for layer in self.num_neurons[1:len(self.num_neurons)]:
                model.add(LSTM(layer, return_sequences=True, activation=self.act_func))
        model.add(TimeDistributed(Dense(self.num_regions, activation='linear')))
        optimizer = optimizers.Adam(lr=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer)
        self.model = model

    def fit(self):
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience),
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
        self.history = self.model.fit(self.train_X, self.train_y, epochs=self.epoch, batch_size=self.batch_size,
                                      validation_data=(self.valid_X, self.valid_y), verbose=2,
                                      shuffle=True, callbacks=callbacks)

    def predict(self, input):
        return self.model.predict(input)

    def save_model(self, model_dir_path, model_id="", region_number=""):
        if (model_id == "") and (region_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
        elif region_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "region_" + str(region_number)
        model_json = self.model.to_json()
        with open(self.predictor_file_path + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.predictor_file_path + ".h5")
        # K.clear_session()

    def load_model(self, model_dir_path, model_id="", region_number=""):
        if (model_id == "") and (region_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
            print("Loading main predictor")
        elif region_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
            print("Loading predictor number {}".format(model_id))
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "region_" + str(region_number)
            print("Loading predictor number {} for region number {}".format(model_id, region_number))
        json_file = open(self.predictor_file_path + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.predictor_file_path + ".h5")
        self.model = loaded_model


class Gru():
    def __init__(self, train_x, train_y, valid_x, valid_y, num_neurons,
                 epoch, batch_size, lr, loss, activation, patience, num_regions):
        # Model properties
        self.num_neurons = num_neurons
        self.epoch = epoch
        self.loss = loss
        self.lr = lr
        self.num_regions = num_regions
        self.batch_size = batch_size
        self.act_func = activation
        self.patience = patience
        self.train_X = train_x
        self.train_y = train_y
        self.valid_X = valid_x
        self.valid_y = valid_y
        self.make_model()

    def make_model(self):
        # design network
        model = Sequential()
        if len(self.num_neurons) == 1:
            model.add(GRU(self.num_neurons[0], input_shape=(None, self.train_X.shape[2]),
                           return_sequences=True, activation=self.act_func))
        else:
            model.add(GRU(self.num_neurons[0], input_shape=(None, self.train_X.shape[2]),
                           return_sequences=True, activation=self.act_func))
            for layer in self.num_neurons[1:len(self.num_neurons)]:
                model.add(GRU(layer, return_sequences=True, activation=self.act_func))
        model.add(TimeDistributed(Dense(self.num_regions, activation='linear')))
        optimizer = optimizers.Adam(lr=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer)
        self.model = model

    def fit(self):
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience),
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
        self.history = self.model.fit(self.train_X, self.train_y, epochs=self.epoch, batch_size=self.batch_size,
                                      validation_data=(self.valid_X, self.valid_y), verbose=2,
                                      shuffle=True, callbacks=callbacks)

    def predict(self, input):
        return self.model.predict(input)

    def save_model(self, model_dir_path, model_id="", region_number=""):
        if (model_id == "") and (region_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
        elif region_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "region_" + str(region_number)
        model_json = self.model.to_json()
        with open(self.predictor_file_path + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.predictor_file_path + ".h5")
        # K.clear_session()

    def load_model(self, model_dir_path, model_id="", region_number=""):
        if (model_id == "") and (region_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
            print("Loading main predictor")
        elif region_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
            print("Loading predictor number {}".format(model_id))
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "region_" + str(region_number)
            print("Loading predictor number {} for region number {}".format(model_id, region_number))
        json_file = open(self.predictor_file_path + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.predictor_file_path + ".h5")
        self.model = loaded_model


class Rnn():
    def __init__(self, train_x, train_y, valid_x, valid_y, num_neurons,
                 epoch, batch_size, lr, loss, activation, patience, num_regions):
        # Model properties
        self.num_neurons = num_neurons
        self.epoch = epoch
        self.loss = loss
        self.lr = lr
        self.num_regions = num_regions
        self.batch_size = batch_size
        self.act_func = activation
        self.patience = patience
        self.train_X = train_x
        self.train_y = train_y
        self.valid_X = valid_x
        self.valid_y = valid_y
        self.make_model()

    def make_model(self):
        # design network
        model = Sequential()
        if len(self.num_neurons) == 1:
            model.add(SimpleRNN(self.num_neurons[0], input_shape=(None, self.train_X.shape[2]),
                                return_sequences=True, activation=self.act_func))
        else:
            model.add(SimpleRNN(self.num_neurons[0], input_shape=(None, self.train_X.shape[2]),
                                return_sequences=True, activation=self.act_func))
            for layer in self.num_neurons[1:len(self.num_neurons)]:
                model.add(SimpleRNN(layer, return_sequences=True, activation=self.act_func))
        model.add(TimeDistributed(Dense(self.num_regions, activation='linear')))
        optimizer = optimizers.Adam(lr=self.lr)
        model.compile(loss=self.loss, optimizer=optimizer)
        self.model = model

    def fit(self):
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience),
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
        self.history = self.model.fit(self.train_X, self.train_y, epochs=self.epoch, batch_size=self.batch_size,
                                      validation_data=(self.valid_X, self.valid_y), verbose=2,
                                      shuffle=True, callbacks=callbacks)

    def predict(self, input):
        return self.model.predict(input)

    def save_model(self, model_dir_path, model_id="", region_number=""):
        if (model_id == "") and (region_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
        elif region_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "region_" + str(region_number)
        model_json = self.model.to_json()
        with open(self.predictor_file_path + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.predictor_file_path + ".h5")
        # K.clear_session()

    def load_model(self, model_dir_path, model_id="", region_number=""):
        if (model_id == "") and (region_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
            print("Loading main predictor")
        elif region_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
            print("Loading predictor number {}".format(model_id))
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "region_" + str(region_number)
            print("Loading predictor number {} for region number {}".format(model_id, region_number))
        json_file = open(self.predictor_file_path + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.predictor_file_path + ".h5")
        self.model = loaded_model


class NonSeq2SeqLstm():
    def __init__(self, train_x, train_y, valid_x, valid_y, num_neurons,
                 epoch, batch_size, lr, loss, activation, patience, num_regions):
        # Model properties
        self.num_neurons = num_neurons
        self.epoch = epoch
        self.loss = loss
        self.lr = lr
        self.num_regions = num_regions
        self.batch_size = batch_size
        self.act_func = activation
        self.patience = patience
        self.train_X = train_x
        self.train_y = train_y
        self.valid_X = valid_x
        self.valid_y = valid_y
        self.make_model()

    def make_model(self):
        # design network
        model = Sequential()
        if len(self.num_neurons) == 1:
            model.add(LSTM(self.num_neurons[0], input_shape=(None, self.train_X.shape[2]),
                           return_sequences=False, activation=self.act_func))
        else:
            model.add(LSTM(self.num_neurons[0], input_shape=(None, self.train_X.shape[2]),
                           return_sequences=True, activation=self.act_func))
            for layer in self.num_neurons[1:len(self.num_neurons)-1]:
                model.add(LSTM(layer, return_sequences=True, activation=self.act_func))
            model.add(LSTM(self.num_neurons[-1], return_sequences=False, activation=self.act_func))
        model.add(Dense(self.num_regions, activation='linear'))
        optimizer = optimizers.adagrad()
        model.compile(loss=self.loss, optimizer=optimizer)
        self.model = model

    def fit(self):
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience),
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
        self.history = self.model.fit(self.train_X, self.train_y, epochs=self.epoch, batch_size=self.batch_size,
                                      validation_data=(self.valid_X, self.valid_y), verbose=2,
                                      shuffle=True, callbacks=callbacks)

    def predict(self, input):
        return self.model.predict(input)

    def save_model(self, model_dir_path, model_id="", region_number=""):
        if (model_id == "") and (region_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
        elif region_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "region_" + str(region_number)
        model_json = self.model.to_json()
        with open(self.predictor_file_path + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.predictor_file_path + ".h5")
        # K.clear_session()

    def load_model(self, model_dir_path, model_id="", region_number=""):
        if (model_id == "") and (region_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
            print("Loading main predictor")
        elif region_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
            print("Loading predictor number {}".format(model_id))
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "region_" + str(region_number)
            print("Loading predictor number {} for region number {}".format(model_id, region_number))
        json_file = open(self.predictor_file_path + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.predictor_file_path + ".h5")
        self.model = loaded_model


class DNN():
    def __init__(self, train_x, train_y, valid_x, valid_y, output_shape,
                 num_neurons, epoch, batch_size, lr, loss, activation, patience):
        #Model properties
        self.num_neurons = num_neurons
        self.epoch = epoch
        self.loss = loss
        self.lr = lr
        self.batch_size = batch_size
        self.act_func = activation
        self.patience = patience
        self.train_X = train_x
        self.train_y = train_y
        self.valid_X = valid_x
        self.valid_y = valid_y
        self.output_shape = output_shape
        self.make_model()

    def make_model(self):
        # design network
        model = Sequential()
        if len(self.num_neurons) == 0:
            model.add(Dense(self.output_shape, activation='linear', input_shape=(self.train_X.shape[1],)))
        elif len(self.num_neurons) == 1:
            model.add(Dense(self.num_neurons[0], input_shape=(self.train_X.shape[1],),
                            activation=self.act_func))
            model.add(Dense(self.output_shape, activation='linear'))
        else:
            model.add(Dense(self.num_neurons[0], input_shape=(self.train_X.shape[1],),
                            activation=self.act_func))
            for layer in self.num_neurons[1:]:
                model.add(Dense(layer, activation=self.act_func))
            model.add(Dense(self.output_shape, activation='linear'))
        adam = optimizers.adam(lr=self.lr)
        model.compile(loss=self.loss, optimizer=adam)
        self.model = model

    def fit(self):
        callbacks = [EarlyStopping(monitor='val_loss', patience=self.patience),
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
        self.history = self.model.fit(self.train_X, self.train_y, epochs=self.epoch, batch_size=self.batch_size,
                                      validation_data=(self.valid_X, self.valid_y), verbose=2,
                                      shuffle=True, callbacks=callbacks)

    def predict(self, input):
        return self.model.predict(input)

    def save_model(self, model_dir_path, model_id="", region_number=""):
        if (model_id == "") and (region_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
        elif region_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "region_" + str(region_number)
        model_json = self.model.to_json()
        with open(self.predictor_file_path + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.predictor_file_path + ".h5")
        K.clear_session()

    def load_model(self, model_dir_path, model_id="", region_number=""):
        if (model_id == "") and (region_number == ""):
            self.predictor_file_path = model_dir_path + "/main_predictor"
            print("Loading main predictor")
        elif region_number == "":
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id)
            print("Loading predictor number {}".format(model_id))
        else:
            self.predictor_file_path = model_dir_path + "/predictor_" + str(model_id) + "region_" + str(region_number)
            print("Loading predictor number {} for region number {}".format(model_id, region_number))
        json_file = open(self.predictor_file_path + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.predictor_file_path + ".h5")
        self.model = loaded_model


def add_preds(data, preds, label, regions):
    for region in regions:
        col_name = label + "_" + str(region)
        data[col_name] = None
    for i in range(len(regions), 0, -1):
        data[data.columns[-i]] = preds[:, len(regions)-i]
    return data


def denormalize_data(data, labels, mean, std):
    all_labels = []
    for label in labels:
        all_labels += [col for col in data.columns if col.startswith(label)]
    for label in all_labels:
        data[label] = data[label].apply(lambda x: denormalize(x, mean, std))
    return data


def round_values(data, labels):
    all_labels = []
    for label in labels:
        all_labels += [col for col in data.columns if col.startswith(label)]
    for label in all_labels:
        data[label] = data[label].apply(lambda x: round(x))
    return data


def predictions_to_csv(data, model_dir_path, file_name):
    data.to_csv(model_dir_path + "/" + file_name + "_preds.csv", index=False)


def calculate_sma(list, lookback):
    sma = list.rolling(lookback).mean()
    sma = sma.shift(1)
    return sma


def calculate_dema(list, lookback):
    ema = list.ewm(span=lookback, adjust=False, min_periods=lookback).mean()
    ema_ema = ema.ewm(span=lookback, adjust=False, min_periods=lookback).mean()
    dema = 2 * ema - ema_ema
    dema = dema.shift(1)
    return dema

def calculate_ema(list, lookback):
    ema = list.ewm(span=lookback, adjust=False, min_periods=lookback).mean()
    ema = ema.shift(1)
    return ema

