import numpy as np
import pickle

class AnomalyDetector:

    def __init__(self):
        pass

    def diagnostic_model(df):

        def load_models(de_model_path = r'D:\Repositories\msc-thesis\arima_model_de.pkl',
                        fe_model_path = r'D:\Repositories\msc-thesis\arima_model_fe.pkl'):
            with open(de_model_path, 'rb') as pkl_file:
                de_model = pickle.load(pkl_file)
            with open(fe_model_path, 'rb') as pkl_file:
                fe_model = pickle.load(pkl_file)

            return de_model, fe_model
        
        def anomaly_detection(df, de_model, fe_model):
            
            best_thr = 0.013882980550933777
            de_signal = df['DE'].reset_index(drop = True)
            fe_signal = df['FE'].reset_index(drop = True)

            de_pred = de_model.predict(start=de_signal.index[0], end=de_signal.index[-1], dynamic = False)
            fe_pred = fe_model.predict(start=fe_signal.index[0], end=fe_signal.index[-1], dynamic = False)

            mse_de = np.mean((de_signal.values.reshape(-1, 1) - de_pred.values.reshape(-1, 1)) ** 2, axis = 1)
            mse_fe = np.mean((fe_signal.values.reshape(-1, 1) - fe_pred.values.reshape(-1, 1)) ** 2, axis = 1)
            avg_mse = (mse_de + mse_fe) / 2
            df['anomaly'] = np.where(avg_mse >= best_thr, 1, 0)

            return df

        de_model, fe_model = load_models()
        df = anomaly_detection(df, de_model, fe_model)
        anom_df = df[df['anomaly'] == 1]

        return anom_df