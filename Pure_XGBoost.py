import pandas as pd
import datetime
import numpy as np
import util
import pickle
from xgboost import XGBRegressor
import gc
import time

pd.options.mode.chained_assignment = None


# Set model construction or inference mode
infer = True
model_number = 1

# Props
grid = 16
agg_interval = 15
lookBack = 4
data_file_path = "./Data/ridereqs-" + str(grid) + "X" + str(grid) + "-" + str(agg_interval) + "min.csv"
train_start_date = datetime.datetime(2017, 9, 1)
train_end_date = datetime.datetime(2017, 11, 11)
val_start_date = datetime.datetime(2017, 11, 11)
val_end_date = datetime.datetime(2017, 12, 1)
test_start_date = datetime.datetime(2017, 12, 1)
test_end_date = datetime.datetime(2017, 12, 21)
per_day_limit = 0

# XGB predictor props
learning_rate = 0.01
num_predictors = 1000
max_depth = 10
sampling_proportion = 0.8
colsample_bytree = 0.5

# make model directory
if not infer:
    model_number = 1
    model_dir_path = util.make_model_dir("./models/thesis/XGB", "Pure_XGBoost")
else:
    model_dir_path = "./models/thesis/XGB/model-" + str(model_number)

# Load data
dataset = util.GenrateOtherMethodsData(data_file_path=data_file_path, lookBack=lookBack, interval=agg_interval,
                                       train_start_date=train_start_date, train_end_date=train_end_date,
                                       val_start_date=val_start_date, val_end_date=val_end_date,
                                       test_start_date=test_start_date, test_end_date=test_end_date,
                                       per_day_limit=per_day_limit, normalize=True, input_3d=False)
# Training Phase
if not infer:
    t = time.time()
    xgbs = []
    for i in range(len(dataset.regions)):
        print("Training XGB number {}".format(i+1))
        xgb = XGBRegressor(learning_rate=learning_rate, n_estimators=num_predictors, max_depth=max_depth,
                           nthread=-1, subsample=sampling_proportion, colsample_bytree=colsample_bytree)
        xgb.fit(dataset.train_X, dataset.train_y[:, i])
        xgbs.append(xgb)
        with open(model_dir_path + "/XGB_region_{}.pkl".format(i+1), 'wb') as f:
            pickle.dump(xgbs[-1], f)
        gc.collect()
    elapsed = time.time() - t
    print('Elapsed: {}'.format(elapsed))



# Inference Phase
else:
    xgbs = []
    # Load models
    for i in range(len(dataset.regions)):
        with open(model_dir_path + "/XGB_region_{}.pkl".format(i+1), 'rb') as f:
            unpickler = pickle.Unpickler(f)
            xgbs.append(unpickler.load())
    # Predict
    main_y_hat = np.zeros((dataset.test_X.shape[0], len(dataset.regions)))
    for i in range(len(xgbs)):
        print("Predicting XGB number {}".format(i + 1))
        main_y_hat[:, i] = np.array(xgbs[i].predict(dataset.test_X)).flatten()

    # Add predictions to DataFrame
    dataset.test = util.add_preds(dataset.test, main_y_hat, "XGB", dataset.regions)

    # De-normalize data
    dataset.test = util.denormalize_data(dataset.test, ['target', 'XGB'], dataset.mean, dataset.std)

    # # Round predictions
    # dataset.test = util.round_values(dataset.test, ['XGB'])

    # Calculate Errors
    util.calculate_errors(dataset.test, target="target", pred="XGB")

    # Write results out
    util.result_file(dataset.test, model_path=model_dir_path, target="target", model_names=["XGB"])

    # Write predicted data out
    util.predictions_to_csv(dataset.test, model_dir_path=model_dir_path, file_name="XGB")
