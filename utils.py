import numpy as np
from scipy import stats


def remove_outlier_rows(df, col, z_max=3):
    while True:
        z_mag = np.abs(stats.zscore(df[col]))
        idxmax = z_mag.idxmax()
        if z_mag[idxmax]<z_max: break  # if good, break before drop
        df = df.drop(idxmax)
    return df.copy()


def find_closest_row(value, df, colname, use_next=False):
    """ Finds the closest previous row."""
    # Fastest way to find the neighboring rows
    next_row = np.searchsorted(df[colname].values, value)
    # If the next row is the first row, just return the first row
    if next_row==0:
        return df.iloc[next_row]
    # Finding the previous row
    prev_row = (next_row-1).clip(0)
    return df.iloc[next_row] if use_next else df.iloc[prev_row]


def slice_between(df, col, low, high):
    return df[df[col].between(low, high)].copy()
