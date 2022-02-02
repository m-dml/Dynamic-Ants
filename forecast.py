import multiprocessing as mp
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas import DataFrame
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EnergyDataset

global model


def do(x, counter):
    x = x.squeeze(0).cpu().numpy()
    preds = model.forecast(x, steps=168)[:, 1:]
    return counter, preds


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weights_path", type=str, default=".", help="Model weights path")
    parser.add_argument("--save_dir", type=str, help="Directory where weights and results are saved", default=".")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the data you want to predict",
        default="/hkfs/work/workspace/scratch/bh6321-energy_challenge/data",
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    data_dir = args.data_dir

    # load model with pretrained weights
    weights_path = os.path.join(args.weights_path, "var_model.weights")
    if not os.path.exists(weights_path):
        raise FileExistsError(f"The file for the trained model does not exist: {weights_path}")

    model = VARResultsWrapper.load(weights_path)

    # dataloader
    test_file = os.path.join(data_dir, "test.csv")
    valid_file = os.path.join(data_dir, "valid.csv")
    data_file = test_file if os.path.exists(test_file) else valid_file
    df = pd.read_csv(data_file)

    city_names = pd.unique(df["City"])

    valid_dataset = EnergyDataset(
        data_file=data_file, historic_window=7 * 24, forecast_horizon=7 * 24, city_names=city_names
    )

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, prefetch_factor=2)

    # run inference
    process_count = mp.cpu_count()
    print("Number of processes found: ", process_count)

    tqdm._instances.clear()
    with Parallel(n_jobs=process_count - 8) as parallel:
        results = parallel(delayed(do)(x, counter) for counter, (x, _) in enumerate(tqdm(valid_dataloader, position=0)))

    sorted_predictions = np.array(
        list(
            pd.DataFrame(results, columns=["counter", "arrays"])
            .sort_values(by="counter")["arrays"]
            .apply(lambda x: np.array(x))
        )
    )
    sorted_predictions = np.moveaxis(sorted_predictions, -1, 0)

    final_predictions = []
    for city in sorted_predictions:
        for prediction in city:
            final_predictions.append(prediction)

    df = DataFrame(final_predictions)

    # save to csv
    result_path = os.path.join(save_dir, "forecasts.csv")
    df.to_csv(result_path, header=False, index=False)

    print(f"Done! The result is saved in {result_path}")
