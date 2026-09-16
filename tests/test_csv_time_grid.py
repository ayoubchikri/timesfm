# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Missing observations must retain their original time positions."""

import importlib.util
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "timesfm-forecasting/scripts/forecast_csv.py"
spec = importlib.util.spec_from_file_location("forecast_csv", SCRIPT)
forecast_csv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forecast_csv)



def test_missing_observations_keep_shared_time_grid(tmp_path):
  frame = pd.DataFrame({
    "date": pd.date_range("2026-01-01", periods=5),
    "sales": [1.0, np.nan, 3.0, 4.0, np.nan],
    "demand": [np.nan, 2.0, 3.0, np.nan, 5.0],
    "empty": [np.nan] * 5,
  })
  original = frame.copy(deep=True)
  model = mock.Mock()
  model.forecast.return_value = (np.zeros((3, 2)), np.zeros((3, 2, 10)))

  results = forecast_csv.forecast_series(model, frame, ["sales", "demand", "empty"], 2)

  inputs = model.forecast.call_args.kwargs["inputs"]
  for column, values in zip(["sales", "demand", "empty"], inputs):
    assert values.dtype == np.float32
    assert values.shape == (len(frame),)
    np.testing.assert_array_equal(values, frame[column].to_numpy(dtype=np.float32))
  pd.testing.assert_frame_equal(frame, original)

  output_path = tmp_path / "forecast.csv"
  forecast_csv.write_csv_output(results, str(output_path), frame, "date", 2)
  exported = pd.read_csv(output_path, parse_dates=["date"])
  for _, group in exported.groupby("series"):
    assert group["date"].tolist() == list(pd.date_range("2026-01-06", periods=2))
