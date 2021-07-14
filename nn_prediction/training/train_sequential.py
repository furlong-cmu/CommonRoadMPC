import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plt
from globals import *
from nn_prediction.training.util import *
import json


def input_output_split(data, num_history_steps, in_cols, out_cols, fraction_training=0.9):
    '''
    TODO: Filter out resets.
    '''
    num_rows, num_cols = data.shape
    train_split = int(fraction_training * num_rows)
    num_train_batches = train_split - num_history_steps

    train_xs = np.zeros((num_train_batches,
                       num_history_steps,
                       len(in_cols)))
    train_ys = np.zeros((num_train_batches, 
                         1,
                         len(out_cols)))
    test_xs = np.zeros((num_rows-train_split,
                        num_history_steps,
                        len(in_cols)))
    test_ys = np.zeros((num_rows-train_split,
                        1, len(out_cols)))
                    

    for i in range(num_history_steps,train_split-1):
        batch_data = data[i-num_history_steps:i,in_cols]
        train_xs[i-num_history_steps,:,:] = batch_data
        train_ys[i-num_history_steps,:,:] = data[i+1,out_cols] - data[i,out_cols]

    for i in range(train_split,num_rows-1):
        test_xs[i-train_split,:,:] = data[i-num_history_steps:i,in_cols]
        test_ys[i-train_split,:,:] = data[i+1,out_cols] - data[i,out_cols]


    return train_xs, train_ys, test_xs, test_ys
    # Making the LSTM stateful

def train_network():
    '''
    Trying to learn the function \dot{x} = f(x_{t,t-k}, u_{t})
    Commands are  steering, throttle, break, and reverse
    State is (x_vel, y_vel, speed, x_accel, y_accel, steering angle, body angle, yaw_rate, drift_angle)
    Prediction is change in (x, y, x_vel, y_vel, speed, x_accel, y_accel, steering angle, body angle, yaw_rate, drift_angle)
    
    First version of the network will learn 1 time step in
    the future. 
    '''


    nn_settings ={
        "predict_delta" : PREDICT_DELTA,
        "normalize_data": NORMALITE_DATA,
        "model_name": MODEL_NAME
    }
 

    # load the dataset
    # 0: time,
    # 1: command.autodrive_enabled, <- IGNORE
    # 2: command.steering, <- u1
    # 3: command.throttle, <- u2
    # 4: command.brake, <- u3?
    # 5: command.reverse, <- filter out where non-zero.
    # 6: position_m.x,
    # 7: position_m.y,
    # 8: velocity_m_per_sec.x,
    # 9: velocity_m_per_sec.y,
    # 10: speed_m_per_sec,
    # 11: accel_m_per_sec_2.x,
    # 12: accel_m_per_sec_2.y,
    # 13: steering_angle_deg,
    # 14: body_angle_deg,
    # 15: yaw_rate_deg_per_sec,
    # 16: drift_angle_deg
#     train_data = np.loadtxt(f'nn_prediction/training_data/{}'.format(TRAINING_DATA_FILE), delimiter=',')
    raw_data = np.loadtxt(f'nn_prediction/training_data/{TRAINING_DATA_FILE}', delimiter=',')
    # filter out reverse
    in_cols = [4,6,7,8,9,10,11,12,13,14,15,16]
    out_cols = [2,3,4,6,7,8,9,10,11,12,13,14,15,16]

    train_xs, train_ys, test_xs, test_ys = input_output_split(raw_data, 5, in_cols, out_cols)
    print('train xs shape: ', train_xs.shape)
    print('train ys shape: ', train_ys.shape)
    print('test xs shape: ', test_xs.shape)
    print('test ys shape: ', test_ys.shape)
    exit()
    non_reverse = raw_data[raw_data[:,5] == 0]
    non_reverse = non_reverse[:, data_cols]  
    (total_rows, total_cols) = non_reverse.shape

    assert int(total_rows) * 0.1 > NUMBER_OF_NEXT_WAYPOINTS, f'10% of data smaller than prediction window'

    train_data = non_reverse[:int(total_rows * 0.9) + NUMBER_OF_NEXT_WAYPOINTS + 1,:]
    validation_data = non_reverse[int(total_rows * 0.9) - NUMBER_OF_NEXT_WAYPOINTS - 1:,:]
    print("train_data.shape", train_data.shape)
    

    # split into input (X) and output (y) variables
    #x1,x2,x3,x4,x5,x6,x7,u1,u2,
    x = train_data[:,1:10]
    #x1n,x2n,x3n,x4n,x5n,x6n,x7n
    y = train_data[:,10:]




    #time, x1,x2,x3,x4,x5,x6,x7,u1,u2
#     validation_data = np.loadtxt('ExperimentRecordings/Dataset-1/Test/Test.csv{}'.format(""), delimiter=',', skiprows=5)
    validation_data = validation_data[2:] #remove header

    #x1,x2,x3,x4,x5,x6,x7,u1,u2,
    x_validation = validation_data[:-1,1:10]
    #x1n,x2n,x3n,x4n,x5n,x6n,x7n
    y_validation = validation_data[1:,1:8]


    #delta is the difference between state and next_state
    delta =  y[:] - x[:,:7]
    delta_validation = y_validation[:] - x_validation[:,:7]

    #if we want to train the network on the state changes instead of the state, use this
    if PREDICT_DELTA:
        y = delta
        y_validation = delta_validation


    # Augmentation for lots of lots of data
    # x, y = augment_data(x,y)

    # Normalize data
    scaler_x = preprocessing.MinMaxScaler().fit(x)
    scaler_y = preprocessing.MinMaxScaler().fit(y)

    if(NORMALITE_DATA):
        x = scaler_x.transform(x)
        y = scaler_y.transform(y)
        x_validation = scaler_x.transform(x_validation)
        y_validation = scaler_y.transform(y_validation)


    # keras model
    model = Sequential()
    model.add(Dense(128, input_dim=9, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(7))


    # compile
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # fit
    history = model.fit(x, y, epochs=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.1,  validation_data=(x_validation, y_validation))

    # Save model and normalization constants
    model_path = 'nn_prediction/models/{}'.format(MODEL_NAME)
    scaler_x_path = 'nn_prediction/models/{}/scaler_x.pkl'.format(MODEL_NAME)
    scaler_y_path = 'nn_prediction/models/{}/scaler_y.pkl'.format(MODEL_NAME)
    nn_settings_path = 'nn_prediction/models/{}/nn_settings.json'.format(MODEL_NAME)

    model.save(model_path)
    joblib.dump(scaler_x, scaler_x_path) 
    joblib.dump(scaler_y, scaler_y_path) 
    with open(nn_settings_path, "w") as outfile:
        outfile.write(json.dumps(nn_settings))


    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('nn_prediction/models/{}/accuracy_curve'.format(MODEL_NAME))

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.yscale('log')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('nn_prediction/models/{}/loss_curve'.format(MODEL_NAME))


    #Evaluate
    _, accuracy = model.evaluate(x, y)
    print('Accuracy: %.2f' % (accuracy*100))






if __name__ == '__main__':

    train_network()
