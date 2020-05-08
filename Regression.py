import pandas as pd
import datetime
import numpy as np
from math import sqrt
import math
import sklearn.metrics as skmets
from sklearn import linear_model
import util
import time
import gc
pd.options.mode.chained_assignment = None

#Model properties
grid = 16
agg_interval = 15
data_file_path = "./Data/ridereqs-" + str(grid) + "X" + str(grid) + "-" + str(agg_interval) + "min.csv"
train_start_date = datetime.datetime(2017, 9, 1)
train_end_date = datetime.datetime(2017, 11, 11)
val_start_date = datetime.datetime(2017, 11, 11)
val_end_date = datetime.datetime(2017, 12, 1)
test_start_date = datetime.datetime(2017, 12, 1)
test_end_date = datetime.datetime(2017, 12, 21)
per_day_limit = 0
lookback = 4

# make model directory
model_number = 1
# model_dir_path = util.make_model_dir("./models/2018/grid-" + str(grid) + "/interval-" +
#                                          str(agg_interval) + '/Regression', "Regression")
model_dir_path = util.make_model_dir("./models/thesis/Regression", "Regression")


def historical_average(dataset):
    ha = dataset.train.copy()
    cols = ['createdAt'] + [col for col in ha.columns if col.startswith('target')]
    ha = ha[cols]
    ha['time'] = ha.createdAt.apply(lambda x: x.time())
    ha['weekday'] = ha.createdAt.apply(datetime.datetime.weekday)
    ha_preds = ha.groupby(['weekday', 'time'], as_index=False).mean()
    ha_preds.columns = ['weekday', 'time'] + ['HA_' + str(x) for x in dataset.regions]
    test = dataset.test.copy()
    test['time'] = test.createdAt.apply(lambda x: x.time())
    test['weekday'] = test.createdAt.apply(datetime.datetime.weekday)
    test = pd.merge(test, ha_preds, how='left', on=['time', 'weekday'])
    test = test.drop(['time','weekday'],axis=1)
    ha_labels = [col for col in test.columns if col.startswith('HA')]
    return np.array(test[ha_labels].values)


def SMA_DEMA(dataset, lookback):
    test = dataset.test.copy()
    for region in dataset.regions:
        target_col = 'target_' + str(region)
        label_sma = 'SMA_' + str(region)
        label_dema = 'DEMA_' + str(region)
        test[label_sma] = util.calculate_sma(test[target_col], lookback)
        test[label_dema] = util.calculate_dema(test[target_col], lookback)
        test.fillna(0, inplace=True)
    sma_labels = [col for col in test.columns if col.startswith('SMA')]
    dema_labels = [col for col in test.columns if col.startswith('DEMA')]
    return np.array(test[sma_labels].values), np.array(test[dema_labels].values)


class LASSO:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.models = []

    def fit(self, data):
        counter = 1
        for region in data.regions:
            print("Fitting region {}/{}".format(counter, len(data.regions)))
            targte_region = 'target_' + str(region)
            train_y = data.train[targte_region].values
            self.models.append(linear_model.Lasso(alpha=self.alpha))
            self.models[-1].fit(data.train_X, train_y)
            counter += 1

    def predict(self, input):
        preds =[]
        for model in self.models:
            preds.append(model.predict(input))
        return np.transpose(np.array(preds))


class RidgeRegression:
    def __init__(self, alpha=0.1, solver='lsqr'):
        self.alpha = alpha
        self.solver = solver
        self.models = []

    def fit(self, data):
        counter = 1
        for region in data.regions:
            print("Fitting region {}/{}".format(counter, len(data.regions)))
            targte_region = 'target_' + str(region)
            train_y = data.train[targte_region].values
            self.models.append(linear_model.Ridge(alpha=self.alpha, solver=self.solver))
            self.models[-1].fit(data.train_X, train_y)
            counter += 1

    def predict(self, input):
        preds = []
        for model in self.models:
            preds.append(model.predict(input))
        return np.transpose(np.array(preds))


class OLSR:
    def __init__(self):
        self.models = []

    def fit(self, data):
        counter = 1
        for region in data.regions:
            print("Fitting region {}/{}".format(counter, len(data.regions)))
            targte_region = 'target_' + str(region)
            train_y = data.train[targte_region].values
            self.models.append(linear_model.Lasso())
            self.models[-1].fit(data.train_X, train_y)
            counter += 1

    def predict(self, input):
        preds = []
        for model in self.models:
            preds.append(model.predict(input))
        return np.transpose(np.array(preds))


# Load data
dataset = util.GenrateOtherMethodsData(data_file_path=data_file_path, lookBack=lookback,
                                       train_start_date=train_start_date, train_end_date=train_end_date,
                                       val_start_date=val_start_date, val_end_date=val_end_date,
                                       test_start_date=test_start_date, test_end_date=test_end_date,
                                       per_day_limit=per_day_limit, normalize=False, input_3d=False, interval=agg_interval)

t = time.time()
ha_preds = historical_average(dataset)
elapsed = time.time() - t
print('Elapsed: {}'.format(elapsed))

t = time.time()
sma_preds, dema_preds = SMA_DEMA(dataset, lookback)
elapsed = time.time() - t
print('Elapsed: {}'.format(elapsed))

# Lasso
lasso = LASSO(alpha=0.2)
t = time.time()
lasso.fit(dataset)
lasso_preds = lasso.predict(dataset.test_X)
elapsed = time.time() - t
print('Elapsed: {}'.format(elapsed))
# OLSR
olsr = OLSR()
t = time.time()
olsr.fit(dataset)
olsr_preds = olsr.predict(dataset.test_X)
elapsed = time.time() - t
print('Elapsed: {}'.format(elapsed))
# Ridge Regression
ridge = RidgeRegression()
t = time.time()
ridge.fit(dataset)
ridge_preds = ridge.predict(dataset.test_X)
elapsed = time.time() - t
print('Elapsed: {}'.format(elapsed))
# Add predictions to DataFrame
dataset.test = util.add_preds(dataset.test, ha_preds, "HA", dataset.regions)
dataset.test = util.add_preds(dataset.test, sma_preds, "SMA", dataset.regions)
dataset.test = util.add_preds(dataset.test, dema_preds, "DEMA", dataset.regions)
dataset.test = util.add_preds(dataset.test, olsr_preds, "OLSR", dataset.regions)
dataset.test = util.add_preds(dataset.test, ridge_preds, "Ridge", dataset.regions)
dataset.test = util.add_preds(dataset.test, lasso_preds, "Lasso", dataset.regions)

# De-normalize data
dataset.test = util.denormalize_data(dataset.test, ['target', 'SMA', 'DEMA', 'OLSR', 'Ridge', 'Lasso'],
                                     dataset.mean, dataset.std)

# Round predictions
dataset.test = util.round_values(dataset.test, ['SMA', 'DEMA', 'OLSR', 'Ridge', 'Lasso'])

# Calculate Errors
util.calculate_errors(dataset.test, target="target", pred="HA")
util.calculate_errors(dataset.test, target="target", pred="SMA")
util.calculate_errors(dataset.test, target="target", pred="DEMA")
util.calculate_errors(dataset.test, target="target", pred="OLSR")
util.calculate_errors(dataset.test, target="target", pred="Ridge")
util.calculate_errors(dataset.test, target="target", pred="Lasso")

# Write results out
util.result_file(dataset.test, model_path=model_dir_path,
                 target="target", model_names=['HA', 'SMA', 'DEMA', 'OLSR', 'Ridge', 'Lasso'])

# Write predicted data out
util.predictions_to_csv(dataset.test, model_dir_path=model_dir_path, file_name="Other_methods")
