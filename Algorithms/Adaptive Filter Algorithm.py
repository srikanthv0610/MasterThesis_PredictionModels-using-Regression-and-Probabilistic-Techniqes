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

    def Kalman_Adaptive_filter(start_len,end_len):

        new_ds = dataframe(start_len, end_len)
        dataset = new_ds.values

        x_compair = dataset[start_len:,1]  ##used for calculating error later
        y_compair = dataset[start_len:,2]

        x_total = dataset[start_len:end_len,1]
        y_total = dataset[start_len:end_len,2]


        ###Prediction state
        x = np.array([[0.0, 0.0]]).T

        ###Initial uncertainity
        P = np.array([[1, 0],
                      [0, 1]])

        ###Dynamic matrix
        dt = 1.0
        A = np.array([[1.0, dt],
                      [0.0, 1.0]])


        ###Measurement Matrix
        H = np.array([[1,0]])

        ###Measurement Noise Covariance
        # R will be updated in each filter step in the adaptive Kalman Filter, here is just the initialization.
        dl = 8
        ra = dl ** 2
        R = np.array([ra])

        ###Process Noise Covariance
        sv = 0.01
        sf = 1
        G = np.array([[0.5 * dt ** 2],
                      [dt]])

        Q = G * G.T * sv ** 2
        esp_max = 4

        ###Identity matrix
        I = np.eye(2)

        #Measuments
        measurements = x_total


        # Preallocation for Plotting
        psx = []
        dxt = []
        Zx = []
        Px = []
        Pdx = []
        Rdx = []
        Kx = []
        Kdx = []
        vx = []
        Pdet = []
        pos_predict = []
        cov_predict = []
        error = []
        su = []
        epsilon_normalize = []

        for i in range(len(measurements)):

            v_x = measurements[i]


            # Time update (Prediction)
            # Project the state ahead
            x_predict = []
            p_predict = []
            x_new = x
            P_new = P
            for j in range(1,21,1):
                sv_new = 0.01
                G_new = np.array([[0.5 * j ** 2],
                              [j]])

                Q_new = G_new * G_new.T * (sv * sf) ** 2

                A_new = np.array([[1.0, j],
                          [0.0, 1.0]])

                x_new = np.dot(A_new,x)

                # Project the error covariance ahead
                P_new = A_new.dot(P).dot(A_new.T) + Q_new

                if j == 1:
                    x = x_new
                    P = P_new

                x_predict.append(float(x_new[0]))
                p_predict.append(float(P_new[0,0]))

            # Measurement update
            # Compute the Kalman gain

            #x = np.dot(A, x)
            #P = A.dot(P).dot(A.T) + Q

            S = H.dot(P).dot(H.T) + R  # S = H * P * H.T + R
            K = P.dot(H.T).dot(np.linalg.pinv(S))



            # Update the estimate via z
            Z = measurements[i]

            y = Z - np.dot(H, x)  # Innovation or Residual

            epsilon = np.dot(y.T, np.linalg.pinv(S)).dot(y)

            #Adaptive adjustment
            sf = 1
            if epsilon > esp_max:
                sf = 1000

            else:
                sf = 1

            # Update the new Estimate
            x = x + np.dot(K, y)

            # Update the error covariance
            P = (I - np.dot(K, H)).dot(P)

            P_det = np.linalg.det(P)

            psx.append(float(x[0]))
            dxt.append(float(x[1]))
            Zx.append(float(Z))
            Px.append(float(P[0, 0]))
            Pdx.append(float(P[1, 1]))
            Rdx.append(float(R))
            Kx.append(float(K[0]))
            Kdx.append(float(K[1]))
            vx.append(float(v_x))
            pos_predict.append(x_predict)
            cov_predict.append(p_predict)
            error.append(float(y))
            Pdet.append(P_det)
            su.append(S)
            epsilon_normalize.append(epsilon)

        pos_predict = np.array(pos_predict)
        pos_predict = pos_predict.T
        #pos_predict2 = pos_predict.astype(int)

        Prediction_error = []
        def Error_cal():

            for row in range(len(pos_predict)):
                Prediction_error_stepwise = []
                for col in range(len(pos_predict[0])):
                    predict_error = pos_predict[row,col] - x_compair[row+col]
                    Prediction_error_stepwise.append(predict_error)
                Prediction_error.append(Prediction_error_stepwise)
        Error_cal()
        Prediction_error = np.array(Prediction_error)
        Prediction_error = Prediction_error.astype(int)

    Kalman_Adaptive_filter(start_len=0, end_len=50000)

if __name__ == '__main__':
    main()