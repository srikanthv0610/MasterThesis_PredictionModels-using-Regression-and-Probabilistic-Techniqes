import numpy as np
import pandas as pd

def main():

    def dataframe(start_len, end_len):

        data_path = 'dataset_ML/Survey_01/Total/TPx_2020-08-18_15-39-19.csv'
        ds = pd.read_csv(data_path, na_values={'/tLeft Screen X': ["not available","n.a."]})
        ds.to_csv("new_ds.csv", index=False, columns=['Timestamp', '\tLeft Screen X', ' Left Screen Y', ' Left Blink'])
        new_ds = pd.read_csv('new_ds.csv')
        new_ds['Time'] = (new_ds.Timestamp - new_ds.Timestamp[0])
        new_ds = new_ds.drop(columns=['Timestamp'])
        new_ds = new_ds[['Time', '\tLeft Screen X', ' Left Screen Y', ' Left Blink']]
        #Check for blank spaces and replace them with the previous position
        for i in range (end_len):
            if (new_ds.iloc[i,3]) == 1:
                new_ds.iloc[i, 1] = new_ds.iloc[i-1, 1]
                new_ds.iloc[i, 2] = new_ds.iloc[i-1, 2]

        return (new_ds)

    def lin_regressor(start_len, end_len, train_len, predict_len):
        new_ds = dataframe(start_len, end_len)
        dataset = new_ds.values

        Time = new_ds.iloc[:, :1].values
        Left_X = dataset[:, 1]
        Left_Y = dataset[:, 2]

        X = Time.reshape(-1, 1)  # -1 tells numpy to figure out the dimension by itself
        ones = np.ones([X.shape[0], 1])  # create a array containing only ones
        X = np.concatenate([ones, X], 1)

        # print(np.array(X[:,1]))
        y1 = Left_X.reshape(-1, 1)
        y2 = Left_Y.reshape(-1, 1)

        Cost_function_X = []
        Cost_function_Y = []
        Prediction_ErrorX = []
        Prediction_ErrorY = []

        predict_X = []
        predict_Y = []

        for i in range(start_len, end_len, 1):
            # Set the hyper parameters:
            alpha = 0.001
            iters = 10

            t0 = y1[i]  # get y_intercept
            t1 = y2[i]

            thetaX = np.array([[float(t0), 0]])
            thetaY = np.array([[float(t1), 0]])

            # theta = np.array([[1.0, 1.0]])
            X_train = X[i:i + train_len]
            y1_train = y1[i:i + train_len]
            y2_train = y2[i:i + train_len]

            #Regression parameter calculation
            theta_LeftX, cost1 = gradientDescent(X_train, y1_train, thetaX, alpha, iters)
            theta_LeftY, cost2 = gradientDescent(X_train, y2_train, thetaY, alpha, iters)

            Cost_function_X.append(cost1)
            Cost_function_Y.append(cost2)

            # Using Hypotheis to predict
            LeftX_predict = theta_LeftX[:, 0] + ((theta_LeftX[:, 1]) * Time[i + train_len: i + train_len + predict_len])
            LeftY_predict = theta_LeftY[:, 0] + ((theta_LeftY[:, 1]) * Time[i + train_len: i + train_len + predict_len])

            predict_X.append(LeftX_predict)
            predict_Y.append(LeftY_predict)

            Error_X = LeftX_predict - y1[i + train_len: i + train_len + predict_len]
            Error_Y = LeftY_predict - y2[i + train_len: i + train_len + predict_len]
            Prediction_ErrorX.append(Error_X)
            Prediction_ErrorY.append(Error_Y)


        # Converting 3D to 2D array
        Prediction_ErrorX = np.reshape(Prediction_ErrorX, (len(Prediction_ErrorX), len(Prediction_ErrorX[0])))
        Prediction_ErrorY = np.reshape(Prediction_ErrorY, (len(Prediction_ErrorY), len(Prediction_ErrorY[0])))

        Prediction_ErrorX = np.array(Prediction_ErrorX)
        Prediction_ErrorY = np.array(Prediction_ErrorY)

        predict_X = np.array(predict_X)
        predict_Y = np.array(predict_Y)

        #Final Predictions: each row contains the 20 predictions
        predict_X = np.reshape([predict_X], [end_len,20])
        predict_Y = np.reshape([predict_Y], [end_len,20])


        ErrorX_termwise = []
        ErrorY_termwise = []
        for j in range(0, predict_len, 1):
            n_termX = [i[j] for i in Prediction_ErrorX]  # Get 1st, 2nd.....,nth terms together
            n_termY = [i[j] for i in Prediction_ErrorY]

            ErrorX_termwise.append(n_termX)
            ErrorY_termwise.append(n_termY)


        ErrorX_termwise = np.array(ErrorX_termwise)
        ErrorY_termwise = np.array(ErrorY_termwise)
        ErrorX_termwise = ErrorX_termwise.astype(int)
        ErrorY_termwise = ErrorY_termwise.astype(int)

    #Compute Cost Function
    def computeCost(X, y, theta):
        return np.sum(np.power(((X @ theta.T) - y),2)) / (2 * len(X))
        # @ means matrix multiplication of arrays. If we want to use * for multiplication we will have to convert all arrays to matrices

    #Compute Gradient Descent
    def gradientDescent(X, y, theta, alpha, iters):
        for i in range(iters):
            theta = theta - (alpha / len(X)) * np.sum((X @ theta.T - y) * X, axis=0)
            cost = computeCost(X, y, theta)
        return (theta, cost)

    lin_regressor(start_len=0000, end_len=50000, train_len=5, predict_len=20)


if __name__ == '__main__':
    main()