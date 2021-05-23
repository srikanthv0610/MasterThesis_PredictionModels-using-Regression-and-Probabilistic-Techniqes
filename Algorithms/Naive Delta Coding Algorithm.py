import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

def main():

    def dataframe(start_len, end_len):
        plt.style.use('fivethirtyeight')

        #Replace the CSV file path here
        data_path = 'dataset_ML/Survey_01/Total/TPx_2020-08-27_14-04-55.csv'
        ds = pd.read_csv(data_path, na_values={'/tLeft Screen X': ["not available","n.a."]})

        ds.to_csv("new_ds.csv", index=False, columns=['Timestamp', '\tLeft Screen X', ' Left Screen Y', ' Left Blink'])
        new_ds = pd.read_csv('new_ds.csv')
        new_ds['Time'] = (new_ds.Timestamp - new_ds.Timestamp[0])
        new_ds = new_ds.drop(columns=['Timestamp'])
        new_ds = new_ds[['Time', '\tLeft Screen X', ' Left Screen Y', ' Left Blink']]
        #Check for blank spaces and replace them with the previous position
        for i in range (start_len, end_len, 1):
            if (new_ds.iloc[i,3]) == 1:
                new_ds.iloc[i, 1] = new_ds.iloc[i-1, 1]
                new_ds.iloc[i, 2] = new_ds.iloc[i-1, 2]

        return (new_ds)

    def delta_coding(start_len, end_len, predict_len):
        new_ds = dataframe(start_len,end_len)
        dataset = new_ds.values

        x_compair = dataset[start_len:, 1]  ##used for calculating error later
        y_compair = dataset[start_len:, 2]


        Prediction_ErrorX = []
        Prediction_ErrorY = []
        predict_X = []
        predict_Y = []

        for i in range(start_len, end_len, 1):
            x_predict = x_compair[i]
            y_predict = y_compair[i]

            X_Err = []
            Y_Err = []
            for j in range(1, predict_len+1,1):

                x_error = x_compair[i+j] - x_predict
                y_error = y_compair[i + j] - y_predict
                predict_X.append(x_predict)
                predict_Y.append(y_predict)
                X_Err.append(x_error)
                Y_Err.append(y_error)

            Prediction_ErrorX.append(X_Err)
            Prediction_ErrorY.append(Y_Err)

        #Error
        Prediction_ErrorX = np.array(Prediction_ErrorX)
        Prediction_ErrorY = np.array(Prediction_ErrorY)


        predict_X = np.array(predict_X)
        predict_Y = np.array(predict_Y)

        #Final Predictions: each row contains the 20 predictions
        predict_X = np.reshape([predict_X], [end_len,20])
        predict_Y = np.reshape([predict_Y], [end_len,20])


    delta_coding(start_len=0000, end_len=20000, predict_len=20)



if __name__ == '__main__':
    main()