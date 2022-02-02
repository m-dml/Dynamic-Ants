import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class EnergyDataset(Dataset):
    def __init__(self, data_file, historic_window, forecast_horizon, city_names):
        # Input sequence length and output (forecast) sequence length
        self.historic_window = historic_window
        self.forecast_horizon = forecast_horizon

        df = pd.read_csv(data_file)
        # Load Data from csv to Pandas Dataframe
        city_data = []
        for city in city_names:
            city_data.append(df[df["City"] == city]["Load [MWh]"].to_numpy())

        unique_time = np.expand_dims(
            pd.to_datetime(df[df["City"] == city]["Time [s]"]).view(np.int64).to_numpy() / 10 ** 9 / 3600, 1
        )
        city_data = np.moveaxis(np.array(city_data), 0, -1)
        self.dataset = city_data
        self.dataset = np.concatenate([unique_time, self.dataset], axis=1)

    def __len__(self):
        return int(self.dataset.shape[0] - self.historic_window - self.forecast_horizon)

    def __getitem__(self, idx):
        # translate idx (day nr) to array index
        x = self.dataset[
            idx: idx + self.historic_window,
        ]
        y = self.dataset[
            idx + self.historic_window: idx + self.historic_window + self.forecast_horizon,
        ]

        return x, y
