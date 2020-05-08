import pandas as pd
import datetime
import numpy as np
import math
import util
import time
from keras.callbacks import CSVLogger
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, TimeDistributed, SimpleRNN, GRU, CuDNNGRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from sklearn.metrics import mean_squared_error
from keras.layers import Dense
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
from keras import backend as K

pd.options.mode.chained_assignment = None
correct_error = 2
correct_percentage = 10

# Props.
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

# Main predictor props.
main_epoch = 500
main_batch_size = 20
main_lr = 0.0001
main_loss = util.mse_loss_mean
main_activation = "tanh"
main_patience = 5


dataset = pd.read_csv(data_file_path)
dataset['createdAt'] = pd.to_datetime(dataset['createdAt'], format='%Y-%m-%d %H:%M:%S')
days = (dataset.createdAt.max().date() - dataset.createdAt.min().date()).days
df = dataset.groupby(['region'], as_index=False).agg({'ride_count': {"average": ['sum']}})
df.columns = df.columns.droplevel(1)
df.average = df.average / days
# regions = df[df.average >= per_day_limit].region.values
regions = [16 * a + b for a in range(7, 13) for b in range(7, 13)]
dataset = dataset[dataset.region.isin(regions)]
dataset = dataset[['createdAt', 'region', 'ride_count']]
dataset = dataset[dataset.createdAt >= train_start_date]
regions = dataset.region.unique()
mean = dataset.ride_count.mean()
std = dataset.ride_count.std()
dataset = dataset.groupby('region', as_index=False)


targets = pd.DataFrame()
targets['createdAt'] = dataset.get_group(regions[0])['createdAt']
targets = pd.DataFrame()
targets['createdAt'] = dataset.get_group(regions[0])['createdAt']
targets['is_holiday'] = targets.createdAt.apply(lambda x: util.is_holiday(x))
targets['is_holiday'] = (targets['is_holiday'] - targets['is_holiday'].min()) / (
    targets['is_holiday'].max() - targets['is_holiday'].min())
# Set Day of week
targets['weekdaynum'] = targets.createdAt.apply(datetime.datetime.weekday)
targets['weekdaynum'] = (targets['weekdaynum'] - targets['weekdaynum'].min()) / (
    targets['weekdaynum'].max() - targets['weekdaynum'].min())
# Set time interval number
targets['timeslot'] = targets.createdAt.apply(lambda x: (x.hour * 2 + x.minute / 30))
targets['sin_time'] = targets.timeslot.apply(lambda x: math.sin(2 * math.pi * x / (24 * (60 / agg_interval))))
targets['cos_time'] = targets.timeslot.apply(lambda x: math.cos(2 * math.pi * x / (24 * (60 / agg_interval))))
targets = targets.drop(['timeslot'], axis=1)
for region in regions:
    df = dataset.get_group(region)[['createdAt', 'ride_count']]
    col_name = 'region_' + str(region)
    df.columns = ['createdAt', col_name]
    targets = pd.merge(targets, df, how='inner', on=['createdAt'])
request_cols = [col for col in targets.columns if col.startswith('region')]
for col in request_cols:
    label = col + "_(t+1)"
    targets[label] = targets[col].shift(-1)
dataset = targets


dataset.dropna(axis=0, how='any', inplace=True)
targets_label = [col for col in dataset.columns if col.startswith('region')]
labels = list(targets_label)
for label in labels:
    dataset[label] = dataset[label].apply(lambda x: util.normalize(x, mean, std))


test = dataset[(dataset['createdAt'] >= test_start_date) & (dataset['createdAt'] < test_end_date)]
val = dataset[(dataset['createdAt'] >= val_start_date) & (dataset['createdAt'] < val_end_date)]
train = dataset[(dataset['createdAt'] >= train_start_date) & (dataset['createdAt'] < train_end_date)]


def split_input_output_reshape(train, test, val, seq_len):
    targets_label = [col for col in dataset.columns[1:] if col.endswith('(t+1)')]
    features_label = [col for col in dataset.columns[1:] if not col.endswith('(t+1)')]
    train_X, train_y = train[features_label].values, train[targets_label].values
    val_X, val_y = val[features_label].values, val[targets_label].values
    test_X, test_y = test[features_label].values, test[targets_label].values
    seq_length = seq_len

    train_X_res = train_X.shape[0] % seq_len
    train_X = train_X[train_X_res:]

    train_y_res = train_y.shape[0] % seq_len
    train_y = train_y[train_y_res:]

    val_X_res = val_X.shape[0] % seq_len
    val_X = val_X[val_X_res:]

    val_y_res = val_y.shape[0] % seq_len
    val_y = val_y[val_y_res:]

    test_X_res = test_X.shape[0] % seq_len
    test_X = test_X[test_X_res:]

    test_y_res = test_y.shape[0] % seq_len
    test_y = test_y[test_y_res:]

    train_X = train_X.reshape((int(train_X.shape[0] / seq_length), seq_length, train_X.shape[1]))
    test_X = test_X.reshape((int(test_X.shape[0] / seq_length), seq_length, test_X.shape[1]))
    val_X = val_X.reshape((int(val_X.shape[0] / seq_length), seq_length, val_X.shape[1]))
    train_y = train_y.reshape((int(train_y.shape[0] / seq_length), seq_length, train_y.shape[1]))
    test_y = test_y.reshape((int(test_y.shape[0] / seq_length), seq_length, test_y.shape[1]))
    val_y = val_y.reshape((int(val_y.shape[0] / seq_length), seq_length, val_y.shape[1]))
    return train_X, train_y, test_X, test_y, val_X, val_y


def train_evaluate(ga_individual_solution):
    # Decode GA solution to integer for window_size and num_units
    window_size_bits = BitArray(ga_individual_solution[0:4])
    first_num_units_bits = BitArray(ga_individual_solution[4:8])
    second_num_units_bits = BitArray(ga_individual_solution[8:])
    window_size = window_size_bits.uint
    first_num_units = first_num_units_bits.uint * 100
    second_num_units = second_num_units_bits.uint * 100
    print('\nWindow Size: ', window_size,
          ', First Num of Units: ', first_num_units,
          ', Second Num of Units: ', second_num_units)

    # Return fitness score of 100 if window_size or num_unit is zero
    if window_size == 0 or first_num_units == 0 or second_num_units == 0:
        return 100,

        # Segment the train_data based on new window_size; split into train and validation (80/20)
    train_X, train_y, test_X, test_y, val_X, val_y = split_input_output_reshape(train, test, val, window_size)

    # Train LSTM model and predict on validation set
    with open('training.log', 'a') as fd:
        fd.write('Window Size:' + str(window_size) + '\n')
        fd.write('First hidden layer:' + str(first_num_units) + '\n')
        fd.write('Second hidden layer:' + str(second_num_units) + '\n')
    model = Sequential()
    model.add(SimpleRNN(first_num_units, input_shape=(None, train_X.shape[2]),
                        return_sequences=True, activation=main_activation))
    model.add(SimpleRNN(second_num_units, return_sequences=True, activation=main_activation))
    model.add(TimeDistributed(Dense(len(regions), activation='linear')))
    optimizer = optimizers.Adam(lr=main_lr)
    model.compile(loss=main_loss, optimizer=optimizer)

    callbacks = [EarlyStopping(monitor='val_loss', patience=main_patience),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
                 CSVLogger('training.log', append=True)]

    history = model.fit(train_X, train_y, epochs=main_epoch, batch_size=main_batch_size,
                        validation_data=(val_X, val_y), verbose=2, shuffle=True, callbacks=callbacks)

    y_pred = model.predict(test_X)

    # Calculate the RMSE score as fitness score for GA
    rmse = np.sqrt(mean_squared_error(np.array(test_y).flatten(), np.array(y_pred).flatten()))
    print('Validation RMSE: ', rmse, '\n')
    with open('training.log', 'a') as fd:
        fd.write('Validation RMSE: ' + str(rmse) + '\n')
    K.clear_session()

    return rmse,

population_size = 60
num_generations = 5
gene_length = 12

# As we are trying to minimize the RMSE score, that's why using -1.0.
# In case, when you want to maximize accuracy for instance, use 1.0
t = time.time()
creator.create('FitnessMax', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n=gene_length)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.5)
toolbox.register('select', tools.selRoulette)
toolbox.register('evaluate', train_evaluate)

population = toolbox.population(n=population_size)
r = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.05, ngen=num_generations, verbose=False)
elapsed = time.time() - t


best_individuals = tools.selBest(population, k=5)
best_window_size = None
best_first_num_units = None
best_second_num_units = None

for bi in best_individuals:
    window_size_bits = BitArray(bi[0:4])
    first_num_units_bits = BitArray(bi[4:8])
    second_num_units_bits = BitArray(bi[8:])
    best_window_size = window_size_bits.uint
    best_first_num_units = first_num_units_bits.uint * 100
    best_second_num_units = second_num_units_bits.uint * 100
    print('\nWindow Size: ', best_window_size,
          ', First Num of Units: ', best_first_num_units,
          ', Second Num of Units: ', best_second_num_units)
    elapsed = time.time() - t
    print('Elapsed: {}'.format(elapsed))
    with open('training.log', 'a') as fd:
        fd.write('Window Size:' + str(best_window_size) + '\n')
        fd.write('First hidden layer:' + str(best_first_num_units) + '\n')
        fd.write('Second hidden layer:' + str(best_second_num_units) + '\n')
with open('training.log', 'a') as fd:
    fd.write('Elapsed: ' + str(elapsed))
