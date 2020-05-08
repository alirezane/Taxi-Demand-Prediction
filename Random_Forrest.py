import pandas as pd
import datetime
import numpy as np
import util
import pickle
import time
from sklearn.ensemble import RandomForestRegressor

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

# Random Forrest predictor props
num_predictors = 1000
max_depth = 10
min_samples_leaf = 15
max_feutures = 0.5

# make model directory
if not infer:
    model_number = 1
    model_dir_path = util.make_model_dir("./models/thesis/RF", "Random_Forrest")
else:
    model_dir_path = "./models/thesis/RF" + "/model-" + str(model_number)

# Load data
dataset = util.GenrateOtherMethodsData(data_file_path=data_file_path, lookBack=lookBack, interval=agg_interval,
                                       train_start_date=train_start_date, train_end_date=train_end_date,
                                       val_start_date=val_start_date, val_end_date=val_end_date,
                                       test_start_date=test_start_date, test_end_date=test_end_date,
                                       per_day_limit=per_day_limit, normalize=True, input_3d=False)
# Training Phase
if not infer:
    random_forests = []
    t = time.time()
    for i in range(len(dataset.regions)):
        print("Training RF number {}/{}".format(i+1, len(dataset.regions)))
        rf = RandomForestRegressor(n_estimators=num_predictors, max_depth=max_depth,
                                   n_jobs=-1, min_samples_leaf=min_samples_leaf, max_features=max_feutures)
        rf.fit(dataset.train_X, dataset.train_y[:, i])
        random_forests.append(rf)
        with open(model_dir_path + "/RF_region_{}.pkl".format(i+1), 'wb') as f:
            pickle.dump(random_forests[-1], f)
    elapsed = time.time() - t
    print('Elapsed: {}'.format(elapsed))


# Inference Phase
else:
    random_forests = []
    # Load models
    for i in range(len(dataset.regions)):
        print("Loading RF model number {}/{}".format(i+1, len(dataset.regions)))
        with open(model_dir_path + "/RF_region_{}.pkl".format(i+1), 'rb') as f:
            unpickler = pickle.Unpickler(f)
            random_forests.append(unpickler.load())
    # Predict
    main_y_hat = np.zeros((dataset.test_X.shape[0], len(dataset.regions)))
    for i in range(len(random_forests)):
        print("Predicting RF number {}".format(i + 1))
        main_y_hat[:, i] = np.array(random_forests[i].predict(dataset.test_X)).flatten()

    # Add predictions to DataFrame
    dataset.test = util.add_preds(dataset.test, main_y_hat, "RF", dataset.regions)

    # De-normalize data
    dataset.test = util.denormalize_data(dataset.test, ['target', 'RF'], dataset.mean, dataset.std)

    # # Round predictions
    # dataset.test = util.round_values(dataset.test, ['RF'])

    # Calculate Errors
    util.calculate_errors(dataset.test, target="target", pred="RF")

    # Write results out
    util.result_file(dataset.test, model_path=model_dir_path, target="target", model_names=["RF"])

    # Write predicted data out
    util.predictions_to_csv(dataset.test, model_dir_path=model_dir_path, file_name="RF")
