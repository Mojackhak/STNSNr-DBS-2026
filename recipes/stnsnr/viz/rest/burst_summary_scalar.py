#%%
import os
from pathlib import Path
import sys

import matplotlib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[4]))
import plugin.default.viz_defaults as cfg

#%% do not show figures during batch processing
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

#%% Project file/folder configuration
project = Path(os.environ["MYLFP_PROJECT_ROOT"])
summary_dir = project / r"summary"
stats_dir = summary_dir / r"stats"
section_dir = stats_dir / r"rest"
burst_dir = section_dir / r"burst" / r"burst"

feature_dirs = [
    burst_dir / r"burst-occupation-scalar",
    burst_dir / r"burst-rate-scalar",
    burst_dir / r"burst-duration-scalar",
    burst_dir / r"burst-mean-scalar",
]

required_columns = {"Band", "Region", "Value"}
region_levels = ["STN", "SNr"]

#%% read data and plot summaries
for wk_dir in feature_dirs:
    wk_dir_name = wk_dir.stem
    param_type = wk_dir_name.rsplit("-", 1)[0]
    data_path = wk_dir / f"{wk_dir_name}_subject_raw.csv"
    save_dir = wk_dir / r"summary"

    print(f"Processing burst summary: {wk_dir_name}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing burst summary input: {data_path}")

    df = pd.read_csv(data_path)
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise KeyError(f"{data_path} is missing required columns: {missing_columns}")

    cfg.plot_categorical_summary_line_wrapper(
        df=df,
        df_type="raw",
        save_dir=save_dir,
        param_type=param_type,
        x_var="Band",
        line_var="Region",
        x_levels=cfg.REST_BURST_BAND_ORDER,
        line_levels=region_levels,
        x_label="Band",
        filebase="Band-Region",
    )

plt.close("all")
