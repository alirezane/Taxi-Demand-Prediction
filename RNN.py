import pandas as pd
import datetime
import util
import time

pd.options.mode.chained_assignment = None

# Set predictor construction or inference mode
infer = True
model_number = 4

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
main_num_neurons = [300, 300]
main_epoch = 500
main_batch_size = 20
main_lr = 0.00001
main_loss = util.mse_loss_mean
main_activation = "tanh"
main_patience = 10

# make model directory
if not infer:
    model_number = 1
    model_dir_path = util.make_model_dir("./models/2017/grid-" + str(grid) + "/interval-" +
                                         str(agg_interval) + '/RNN', "RNN")
    # model_dir_path = util.make_model_dir("./models/thesis/RNN", "RNN")
else:
    model_dir_path = "./models/2017/grid-" + str(grid) + \
                     "/interval-" + str(agg_interval) + "/RNN/model-" + str(model_number)
    # model_dir_path = "./models/thesis/RNN/model-" + str(model_number)

# Load data
dataset = util.GenerateLSTMDate(data_file_path=data_file_path, seq_len=1,
                                train_start_date=train_start_date, train_end_date=train_end_date,
                                val_start_date=val_start_date, val_end_date=val_end_date,
                                test_start_date=test_start_date, test_end_date=test_end_date,
                                per_day_limit=per_day_limit, interval=agg_interval, normalize=True)
# Training Phase
if not infer:
    # Build predictor
    predictor = util.Rnn(train_x=dataset.train_X, train_y=dataset.train_y,
                         valid_x=dataset.val_X, valid_y=dataset.val_y,
                         num_neurons=main_num_neurons, epoch=main_epoch,
                         batch_size=main_batch_size, lr=main_lr, loss=main_loss,
                         activation=main_activation, patience=main_patience, num_regions=len(dataset.regions))

    # Train predictor
    print("Training main predictor")
    t = time.time()
    predictor.fit()
    elapsed = time.time() - t
    print('Elapsed: {}'.format(elapsed))

    # Save trained predictor
    predictor.save_model(model_dir_path)


# Inference Phase
else:
    # Load predictor
    predictor = util.Rnn(train_x=dataset.train_X, train_y=dataset.train_y,
                         valid_x=dataset.val_X, valid_y=dataset.val_y,
                         num_neurons=main_num_neurons, epoch=main_epoch,
                         batch_size=main_batch_size, lr=main_lr, loss=main_loss,
                         activation=main_activation, patience=main_patience, num_regions=len(dataset.regions))
    predictor.load_model(model_dir_path)

    # Predict
    main_y_hat = predictor.predict(dataset.test_X)

    # Add predictions to DataFrame
    main_y_hat = main_y_hat.reshape((main_y_hat.shape[0]*main_y_hat.shape[1], main_y_hat.shape[2]))
    dataset.test = util.add_preds(dataset.test, main_y_hat, "RNN", dataset.regions)

    # De-normalize data
    dataset.test = util.denormalize_data(dataset.test, ['region', 'RNN'], dataset.mean, dataset.std)

    # Round predictions
    dataset.test = util.round_values(dataset.test, ['RNN'])

    # Calculate Errors
    util.calculate_errors(dataset.test, target="region", pred="RNN")

    # Write results out
    util.result_file(dataset.test, model_path=model_dir_path, target="region", model_names=["RNN"])

    # Write predicted data out
    util.predictions_to_csv(dataset.test, model_dir_path=model_dir_path, file_name="RNN")
