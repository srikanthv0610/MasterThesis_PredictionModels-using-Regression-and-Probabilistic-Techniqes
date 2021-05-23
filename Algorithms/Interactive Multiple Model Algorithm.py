import numpy as np
import pandas as pd
import math

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

    def IMM(start_len,end_len):

        new_ds = dataframe(start_len, end_len)
        dataset = new_ds.values

        x_compair = dataset[start_len:,1]  ##used for calculating error later
        y_compair = dataset[start_len:,2]

        x_total = dataset[start_len:end_len,1]
        y_total = dataset[start_len:end_len,2]

        measurements = x_total

        # Number of filters
        N = 2
        dt = 1
        # Velocity model
        x_cv = np.array([[1.0, 0.0, 0.0]]).T

        ###Initial uncertainity
        P_cv = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]])

        ###Measurement Matrix
        H = np.array([[1, 0, 0]])

        ###Measurement Noise Covariance
        # R will be updated in each filter step in the adaptive Kalman Filter, here is just the initialization.
        dl = 8
        ra = dl ** 2
        R = np.array([ra])

        ###Process Noise Covariance
        sv_cv = 0.01

        ###Identity matrix

        I = np.eye(3)

        # Acceleration model
        x_ca = np.array([[1.0, 0.0, 0.0]]).T

        ###Initial uncertainity
        P_ca = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

        ###Process Noise Covariance
        sv_ca = 1

        # Markov chain
        M = np.array([[0.95, 0.05],
                      [0.05, 0.95]])

        # Initial Parameter

        mu = np.array([0.8, 0.2])

        omega = np.zeros((2, 2))

        xbar_cv = x_cv
        xbar_ca = x_ca
        Pbar_cv = P_cv
        Pbar_ca = P_ca
        x_new_cv = x_cv
        P_new_cv = P_cv
        x_new_ca = x_ca
        P_new_ca = P_ca

        psx_cv = []
        dxt_cv = []
        Px_cv = []
        Pdx_cv = []
        Kx_cv = []
        Kdx_cv = []
        vx = []
        pos_predict_cv = []
        cov_predict_cv = []
        error_cv = []
        su_cv = []
        Probability_cv = []

        psx_ca = []
        dxt_ca = []
        acc_ca = []
        Px_ca = []
        Pdx_ca = []
        Kx_ca = []
        Kdx_ca = []
        pos_predict_ca = []
        cov_predict_ca = []
        error_ca = []
        su_ca = []
        pos_est = []
        Probability_ca = []

        for i in range(len(measurements)):

            v_x = measurements[i]

            cbar = np.dot(mu, M)

            for j in range(N):
                for k in range(N):
                    omega[j, k] = (M[j, k] * mu[j]) / cbar[k]

            # Mixing Estimate
            xbar_cv = (omega[0, 0] * x_cv) + (omega[1, 0] * x_cv)
            xbar_ca = (omega[0, 1] * x_ca) + (omega[1, 1] * x_ca)

            Pbar_cv = omega[0, 0] * (P_cv + np.dot((x_cv - xbar_cv), (x_cv - xbar_cv).T)) + \
                      omega[1, 0] + (P_ca + np.dot((x_ca - xbar_ca), (x_ca - xbar_ca).T))

            Pbar_ca = omega[0, 1] * (P_cv + np.dot((x_cv - xbar_cv), (x_cv - xbar_cv).T)) + \
                      omega[1, 1] + (P_ca + np.dot((x_ca - xbar_ca), (x_ca - xbar_ca).T))

            # Time update (Prediction)
            # Project the state ahead
            x_predict_cv = []
            p_predict_cv = []
            x_predict_ca = []
            p_predict_ca = []

            for j in range(1, 21, 1):
                G_cv = np.array([[0.5 * j ** 2],
                                 [j],
                                 [0]])

                Q_cv = G_cv * G_cv.T * sv_cv ** 2

                A_cv = np.array([[1.0, j, 0],
                                 [0.0, 1.0, 0],
                                 [0, 0, 0]])

                x_new_cv = np.dot(A_cv, xbar_cv)

                # Project the error covariance ahead
                P_new_cv = A_cv.dot(Pbar_cv).dot(A_cv.T) + Q_cv

                G_ca = np.array([[0.5 * j ** 2],
                                 [j],
                                 [1]])

                Q_ca = G_ca * G_ca.T * sv_ca ** 2

                A_ca = np.array([[1.0, j, (j ** 2) / 2],
                                 [0.0, 1.0, j],
                                 [0.0, 0.0, 1.0]])

                x_new_ca = np.dot(A_ca, xbar_ca)

                P_new_ca = A_ca.dot(Pbar_ca).dot(A_ca.T) + Q_ca

                if j == 1:
                    x_cv = x_new_cv
                    P_cv = P_new_cv
                    x_ca = x_new_ca
                    P_ca = P_new_ca

                x_predict_cv.append(float(x_new_cv[0]))
                p_predict_cv.append(float(P_new_cv[0, 0]))
                x_predict_ca.append(float(x_new_ca[0]))
                p_predict_ca.append(float(P_new_ca[0, 0]))

            S_cv = H.dot(P_cv).dot(H.T) + R
            S_ca = H.dot(P_ca).dot(H.T) + R

            K_cv = P_cv.dot(H.T).dot(np.linalg.pinv(S_cv))
            K_ca = P_ca.dot(H.T).dot(np.linalg.pinv(S_ca))

            # Update the estimate via z
            Z = measurements[i]

            y_cv = Z - np.dot(H, x_cv)  # Innovation or Residual
            y_ca = Z - np.dot(H, x_ca)

            # Update the new Estimate
            x_cv = x_cv + np.dot(K_cv, y_cv)
            x_ca = x_ca + np.dot(K_ca, y_ca)

            P_cv = (I - np.dot(K_cv, H)).dot(P_cv)
            P_ca = (I - np.dot(K_ca, H)).dot(P_ca)

            # Calculate the Likelyihood
            L_cv = math.exp(-0.5 * (np.dot(y_cv.T, np.linalg.pinv(S_cv)).dot(y_cv))) / (math.sqrt(2 * math.pi * S_cv))

            L_ca = math.exp(-0.5 * (np.dot(y_ca.T, np.linalg.pinv(S_ca)).dot(y_ca))) / (math.sqrt(2 * math.pi * S_ca))

            # Update mu Model Probability
            if L_cv == 0 or L_ca == 0:
                L = np.array([L_cv, L_ca])
                mu = np.array([0.7, 0.3])

            else:

                L_cv = math.log(L_cv)
                L_ca = math.log(L_ca)
                L = np.array([L_cv, L_ca])
                mu = cbar * L
                mu /= sum(mu)

            # print(L_cv, L_ca)
            # print(mu[0], mu[1])

            # IMM Estimate
            x_cv[2] = x_ca[2]

            x = mu[0] * x_cv + mu[1] * x_ca

            P = mu[0] * (np.dot((x_cv - x), (x_cv - x).T) + P_cv) + mu[1] * (np.dot((x_ca - x), (x_ca - x).T) + P_ca)

            x_cv = x_ca = x
            P_cv = P_ca = P

            psx_cv.append(float(x_cv[0]))
            dxt_cv.append(float(x_cv[1]))
            Px_cv.append(float(P_cv[0, 0]))
            Pdx_cv.append(float(P_cv[1, 1]))
            Kx_cv.append(float(K_cv[0]))
            Kdx_cv.append(float(K_cv[1]))
            vx.append(float(v_x))
            pos_predict_cv.append(x_predict_cv)
            cov_predict_cv.append(p_predict_cv)
            error_cv.append(float(y_cv))
            su_cv.append(S_cv)
            Probability_cv.append(mu[0])

            psx_ca.append(float(x_ca[0]))
            dxt_ca.append(float(x_ca[1]))
            acc_ca.append(float(x_ca[2]))
            Px_ca.append(float(P_ca[0, 0]))
            Pdx_ca.append(float(P_ca[1, 1]))
            Kx_ca.append(float(K_ca[0]))
            Kdx_ca.append(float(K_ca[1]))
            pos_predict_ca.append(x_predict_ca)
            cov_predict_ca.append(p_predict_ca)
            error_ca.append(float(y_ca))
            su_ca.append(S_ca)
            # pos_est.append(x)
            Probability_ca.append(mu[1])

        pos_predict_cv = np.array(pos_predict_cv)
        cov_predict_cv = np.array(cov_predict_cv)
        error_cv = np.array(error_cv)
        Kx_cv = np.array(Kx_cv)
        pos_predict_cv = pos_predict_cv.T
        cov_predict_cv = cov_predict_cv.T
        Px_cv = np.array(Px_cv)
        psx_cv = np.array(psx_cv)
        error_estimate_cv = np.sqrt(Px_cv)

        pos_predict_ca = np.array(pos_predict_ca)
        cov_predict_ca = np.array(cov_predict_ca)
        error_ca = np.array(error_ca)
        Kx_ca = np.array(Kx_ca)
        pos_predict_ca = pos_predict_ca.T
        cov_predict_ca = cov_predict_ca.T
        Px_ca = np.array(Px_ca)
        psx_ca = np.array(psx_ca)
        error_estimate_ca = np.sqrt(Px_ca)
        pos_est = np.array(pos_est)
        vx = np.array(vx)
        acc_ca = np.array(acc_ca)

        su_cv = np.array(su_cv)
        su_cv = np.ravel(su_cv)
        su_cv_sqrt = su_cv ** 0.5

        Prediction_error_cv = []
        Prediction_error_ca = []
        Prediction_error_est = []

        def Error_cal():

            for row in range(len(pos_predict_cv)):
                Prediction_error_stepwise_cv = []
                for col in range(len(pos_predict_cv[0])):
                    predict_error_cv = pos_predict_cv[row, col] - x_compair[row + col]
                    Prediction_error_stepwise_cv.append(predict_error_cv)
                Prediction_error_cv.append(Prediction_error_stepwise_cv)

            for row in range(len(pos_est)):
                predict_error_est = pos_est[row] - x_compair[row]
                Prediction_error_est.append(predict_error_est)

        Error_cal()

        Prediction_error_cv = np.array(Prediction_error_cv)
        Prediction_error_cv = Prediction_error_cv.astype(int)

        Prediction_error_est = np.array(Prediction_error_est)
        Prediction_error_est = np.ravel(Prediction_error_est)
        Prediction_error_est = Prediction_error_est.astype(int)

    IMM(start_len=0, end_len=50000)

if __name__ == '__main__':
    main()