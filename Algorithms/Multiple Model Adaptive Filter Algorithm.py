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

    def MMAE(start_len,end_len):

        new_ds = dataframe(start_len, end_len)
        dataset = new_ds.values

        x_compair = dataset[start_len:,1]  ##used for calculating error later
        y_compair = dataset[start_len:,2]

        x_total = dataset[start_len:end_len,1]
        y_total = dataset[start_len:end_len,2]

        measurements = x_total

        ###Prediction state
        x_cv = np.array([[1.0, 0.0]]).T

        ###Initial uncertainity
        P_cv = np.array([[1, 0],
                         [0, 1]])

        ###Measurement Matrix
        H_cv = np.array([[1, 0]])

        ###Measurement Noise Covariance
        # R will be updated in each filter step in the adaptive Kalman Filter, here is just the initialization.
        dl = 8
        ra = dl ** 2
        R = np.array([ra])

        ###Process Noise Covariance
        sv_cv = 0.01

        ###Identity matrix
        I_cv = np.eye(2)

        x_ca = np.array([[1.0, 0.0, 0.0]]).T

        ###Initial uncertainity
        P_ca = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])

        ###Measurement Matrix
        H_ca = np.array([[1, 0, 0]])

        ###Process Noise Covariance
        sv_ca = 1

        ###Identity matrix
        I_ca = np.eye(3)

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

            # Time update (Prediction)
            # Project the state ahead
            x_predict_cv = []
            p_predict_cv = []
            x_predict_ca = []
            p_predict_ca = []

            if i == 0:
                x_new_cv = x_cv
                P_new_cv = P_cv
                x_new_ca = x_ca
                P_new_ca = P_ca

            else:
                x_cv = x_cv_new
                P_cv = P_cv_new
                x_ca = x_ca_new
                P_ca = P_ca_new

            for j in range(1, 21, 1):
                G_cv = np.array([[0.5 * j ** 2],
                                 [j]])

                Q_cv = G_cv * G_cv.T * sv_cv ** 2

                A_cv = np.array([[1.0, j],
                                 [0.0, 1.0]])

                x_new_cv = np.dot(A_cv, x_cv)

                # Project the error covariance ahead
                P_new_cv = A_cv.dot(P_cv).dot(A_cv.T) + Q_cv

                G_ca = np.array([[0.5 * j ** 2],
                                 [j],
                                 [1]])

                Q_ca = G_ca * G_ca.T * sv_ca ** 2

                A_ca = np.array([[1.0, j, (j ** 2) / 2],
                                 [0.0, 1.0, j],
                                 [0.0, 0.0, 1.0]])

                x_new_ca = np.dot(A_ca, x_ca)

                P_new_ca = A_ca.dot(P_ca).dot(A_ca.T) + Q_ca

                if j == 1:
                    x_cv = x_new_cv
                    P_cv = P_new_cv
                    x_ca = x_new_ca
                    P_ca = P_new_ca

                x_predict_cv.append(float(x_new_cv[0]))
                p_predict_cv.append(float(P_new_cv[0, 0]))
                x_predict_ca.append(float(x_new_ca[0]))
                p_predict_ca.append(float(P_new_ca[0, 0]))

            S_cv = H_cv.dot(P_cv).dot(H_cv.T) + R
            S_ca = H_ca.dot(P_ca).dot(H_ca.T) + R

            K_cv = P_cv.dot(H_cv.T).dot(np.linalg.pinv(S_cv))
            K_ca = P_ca.dot(H_ca.T).dot(np.linalg.pinv(S_ca))

            # Update the estimate via z
            Z = measurements[i]

            y_cv = Z - np.dot(H_cv, x_cv)  # Innovation or Residual
            y_ca = Z - np.dot(H_ca, x_ca)

            P_cv = (I_cv - np.dot(K_cv, H_cv)).dot(P_cv)
            P_ca = (I_ca - np.dot(K_ca, H_ca)).dot(P_ca)

            if i == 0:
                L_cv = 0
                L_ca = 0

            else:

                # L_cv = math.exp(0.5 * (np.dot(y_cv.T, np.linalg.pinv(S_cv)).dot(y_cv))) / (math.sqrt(2 * math.pi * S_cv))
                L_cv = np.dot(y_cv.T, np.linalg.pinv(S_cv)).dot(y_cv)

                # L_ca = math.exp(0.5 * (np.dot(y_ca.T, np.linalg.pinv(S_ca)).dot(y_ca))) / (math.sqrt(2 * math.pi * S_ca))
                L_ca = np.dot(y_ca.T, np.linalg.pinv(S_ca)).dot(y_ca)

            if L_cv == 0 or L_ca == 0:
                prob_cv = 0.5
                prob_ca = 0.5

            else:
                den = (L_cv * prob_cv + L_ca * prob_ca)
                prob_cv = L_cv * prob_cv / den
                prob_ca = L_ca * prob_ca / den

            if prob_ca < 0.3:
                prob_ca = 0.3
                prob_cv = 1 - prob_ca

            if prob_cv < 0.3:
                prob_cv = 0.3
                prob_ca = 1 - prob_cv

            # Update the new Estimate
            x_cv_new = x_cv + np.dot(K_cv, y_cv)
            x_ca_new = x_ca + np.dot(K_ca, y_ca)

            x = (prob_cv * x_cv_new[0]) + (prob_ca * x_ca_new[0])

            x_cv_new[0] = x
            x_ca_new[0] = x

            P_cv_new = (P_cv + (x_cv_new - x_cv) * np.transpose(x_cv_new - x_cv))
            P_ca_new = (P_ca + (x_ca_new - x_ca) * np.transpose(x_ca_new - x_ca))

            P = (P_cv_new[0][0] * prob_cv) + (P_ca_new[0][0] * prob_ca)

            P_cv_new[0][0] = P
            P_ca_new[0][0] = P

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
            Probability_cv.append(prob_cv)

            psx_ca.append(float(x_ca[0]))
            dxt_ca.append(float(x_ca[1]))
            Px_ca.append(float(P_ca[0, 0]))
            Pdx_ca.append(float(P_ca[1, 1]))
            Kx_ca.append(float(K_ca[0]))
            Kdx_ca.append(float(K_ca[1]))
            pos_predict_ca.append(x_predict_ca)
            cov_predict_ca.append(p_predict_ca)
            error_ca.append(float(y_ca))
            su_ca.append(S_ca)
            pos_est.append(x)
            Probability_ca.append(prob_ca)

        pos_predict = np.array(pos_predict_cv)
        pos_predict = pos_predict.T

        # pos_predict2 = pos_predict.astype(int)

        Prediction_error = []

        def Error_cal():

            for row in range(len(pos_predict)):
                Prediction_error_stepwise = []
                for col in range(len(pos_predict[0])):
                    predict_error = pos_predict[row, col] - x_compair[row + col]
                    Prediction_error_stepwise.append(predict_error)
                Prediction_error.append(Prediction_error_stepwise)

        Error_cal()
        Prediction_error = np.array(Prediction_error)
        Prediction_error = Prediction_error.astype(int)

    MMAE(start_len=0, end_len=50000)

if __name__ == '__main__':
    main()