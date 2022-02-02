import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="/hkfs/work/workspace/scratch/bh6321-energy_challenge/data/", type=str)
    parser.add_argument("--historic_window", type=int, default=7 * 24, help="input time steps in hours")
    parser.add_argument("--save_dir", default="/hkfs/work/workspace/scratch/bh6321-E1/weights", help="saves the model, if path is provided")
    args = parser.parse_args()

    # Forecast Parameters
    historic_window = args.historic_window

    # Loading Data
    data_dir = args.data_dir
    train_file = os.path.join(data_dir, "train.csv")
    df = pd.read_csv(train_file)

    city_data = []
    city_names = pd.unique(df["City"])

    for city in city_names:
        city_data.append(df[df["City"] == city]["Load [MWh]"].to_numpy())

    unique_time = np.expand_dims(
        pd.to_datetime(df[df["City"] == city]["Time [s]"]).view(np.int64).to_numpy() / 10 ** 9 / 3600, 1
    )
    city_data = np.moveaxis(np.array(city_data), 0, -1)
    train_dataset = city_data

    var_dataset = np.concatenate([unique_time, train_dataset], axis=1)

    # fit the model:
    model = VAR(var_dataset)
    fitted_model = model.fit(maxlags=historic_window)

    fitted_model.save(os.path.join(args.save_dir, "var_model.weights"))


if __name__ == "__main__":
    main()
