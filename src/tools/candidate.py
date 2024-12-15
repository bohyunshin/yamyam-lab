import numpy as np
import os

import pandas as pd
from scipy.spatial import KDTree


DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../data"
)


def get_diner_nearby_candidates(max_distance_km):
    diners = pd.read_csv(os.path.join(DATA_PATH, "diner/diner_df_20241204_yamyam.csv"))
    diner_ids = diners["diner_idx"].unique()
    mapping_diner_idx = {i:id for i,id in enumerate(diner_ids) }

    # Convert latitude and longitude to radians for KDTree
    diner_coords = np.radians([(r[1]["diner_lat"], r[1]["diner_lon"]) for r in diners.iterrows()])

    # Earth's radius in kilometers
    earth_radius_km = 6371

    # Create KDTree
    tree = KDTree(diner_coords)

    # Convert max_distance_km to radians
    max_distance_rad = max_distance_km / earth_radius_km

    # For each of diner, query KDTree for diners within max_distance_rad
    result = {}
    for i,diner_coord in enumerate(diner_coords):
        ref_diner_id = mapping_diner_idx[i]
        # Note: `indices` include referenced diner itself
        indices = tree.query_ball_point(diner_coord, max_distance_rad)
        result[ref_diner_id] = [mapping_diner_idx[id] for id in indices]

    return result
