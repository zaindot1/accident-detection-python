import os
import argparse
import pandas as pd

from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

model = './models/model@1535470106.json'
weights = './models/model@1535470106.h5'
csvfile_path = '../object_detector/results/data/data.csv'
results_dir = '.'

def inference():
    def main():
        df_test = pd.read_csv(csvfile_path)
        x_test = df_test[['scaled_xmin', 'scaled_ymin', 'scaled_xmax', 'scaled_ymax']].values

        scalar = StandardScaler()
        x_test = scalar.fit_transform(x_test)
        scalar.fit_transform((df_test[['scaled_ymax']].values - df_test[['scaled_ymin']]) / 3)

        json_file = open(model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights(weights)
        print("Loaded model from disk")

        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
        distance_pred = loaded_model.predict(x_test)

        distance_pred = scalar.inverse_transform(distance_pred)

        df_result = df_test
        df_result['distance'] = -100000

        for idx, row in df_result.iterrows():
            df_result.at[idx, 'distance'] = distance_pred[idx]
        df_result.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)

    if __name__ == '__main__':
        main()

