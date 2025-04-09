import numpy as np
import pandas as pd

def detect_outliers(
    data: pd.Series,
    n: int = 700,
    m: int = 10,
    k1: float = 1.5,
    k2: float = 1.5,
) -> np.ndarray:
    """
    Detect outliers in a time series using rolling standard deviation and quartile methods.

    Args:
        data: Time series data (Pandas Series or list-like).
        n: Total number of observations to consider (n + m records).
        m: Rolling window size for standard deviation calculation (default: 10).
        k1: Multiplier for lower bound (default: 1.5).
        k2: Multiplier for upper bound (default: 1.5).

    Returns:
        Array of indices where outliers are detected.
    """
    # Convert data to Pandas Series if not already
    series = pd.Series(data)

    # Take the last (n + m) records
    if len(series) > n + m:
        series = series.iloc[-(n + m):]

    # 1. Calculate rolling standard deviation of the last m values
    rolling_std = series.rolling(window=m).std().dropna()

    # 2. Determine reasonable range using quartiles
    Q1 = rolling_std.quantile(0.25)
    Q3 = rolling_std.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - k1 * IQR
    upper_bound = Q3 + k2 * IQR

    # 3. Find outliers (where rolling std is outside bounds)
    outliers_mask = (rolling_std < lower_bound) | (rolling_std > upper_bound)
    outlier_indices = rolling_std[outliers_mask].index.values

    return outlier_indices


# Example Usage:
if __name__ == "__main__":
    use_file = True

    # Data generation
    if use_file:
        df = pd.read_csv("c:\\sources\\fault-detection\\data-file.csv", header=None)
        data_array = df.iloc[:, 0].tolist()
    else:
        data_array = np.random.normal(loc=3, scale=1, size=(700, 1))

    # Detect outliers
    outliers = detect_outliers(data_array, n=700, m=10, k1=1.35, k2=2.25)

    print("Outlier indices:", outliers)
    print("Number of outliers:", len(outliers))