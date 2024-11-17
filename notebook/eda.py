# %%
import os

import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")

# load data
review_1 = pd.read_csv(os.path.join(DATA_PATH, "review_df_20241107_071929_yamyam_1.csv"), index_col=0)
review_2 = pd.read_csv(os.path.join(DATA_PATH, "review_df_20241107_071929_yamyam_2.csv"), index_col=0)
diner = pd.read_csv(os.path.join(DATA_PATH, "diner_df_20241107_071929_yamyam.csv"), index_col=0)

# %%
review_2.loc[review_2["reviewer_user_name"] == "이채은"]
# %%
