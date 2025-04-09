import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional


def analyze_volatility(time_series: Union[List[float], np.ndarray, pd.Series], 
                       n: int = 700, 
                       m: int = 10, 
                       k1: float = 1.5, k2:float=1.5)-> dict:
    """
    Analyzes the volatility of a time series by calculating:
    B = Standard deviation of the rolling standard deviation (volatility of volatility)
    C = Standard deviation of the last m values (recent volatility)
    
    Then determines if the recent volatility (C) is within a reasonable range using quartile methods.
    
    Parameters:
    -----------
    time_series : array-like
        The time series data to analyze
    n : int, default=700
        The number of values to use for calculating standard deviation A
    m : int, default=10
        The number of most recent values to use for calculating standard deviation C
    K1 : float, default=1.5
K2:float,default=1.5
        The multiplier for the interquartile range to define the "reasonable" range
        
    Returns:
    --------
    dict
        A dictionary containing the analysis results
    """
    # Convert input to numpy array for consistent handling
    if isinstance(time_series, pd.Series):
        ts = time_series.values
    else:
        ts = np.array(time_series)
    
    # Check if we have enough data
    if len(ts) < n:
        raise ValueError(f"Time series must have at least {n} values")
    
    # Get the last n values
    last_n_values = ts[-n:]
    
    
    # Calculate rolling standard deviation with window size = m
    rolling_std = []
    for i in range(m-1, n):
        window = last_n_values[i-(m-1):i+1]
        rolling_std.append(np.std(window))
    
    # Calculate B: Standard deviation of the rolling standard deviation
    B = np.std(rolling_std)
    
    # Calculate C: Standard deviation of the last m values
    C = np.std(ts[-m:])
    
    # Calculate quartiles of the rolling standard deviations
    Q1 = np.percentile(rolling_std, 25)
    Q3 = np.percentile(rolling_std, 75)
    IQR = Q3 - Q1
    
    # Define reasonable range
    lower_bound = Q1 - k1 * IQR
    upper_bound = Q3 + k2 * IQR
    
    # Determine if C is within the reasonable range
    is_reasonable = lower_bound <= C <= upper_bound
    
    # Compile results
    results = {
        
        "std_of_rolling_std": B,
        "std_last_m_values": C,
        "quartile_1": Q1,
        "quartile_3": Q3,
        "IQR": IQR,
        "lower_bound": max(0, lower_bound),  # Standard deviation can't be negative
        "upper_bound": upper_bound,
        "is_within_reasonable_range": is_reasonable
    }
    
    return results


if __name__ == "__main__":
    
    use_file = True

    # Data generation
    if use_file:
        df = pd.read_csv("c:\\sources\\fault-detection\\data-file.csv", header=None)
        data_array = df.iloc[:, 0].tolist()
    else:
        data_array = np.random.normal(loc=3, scale=1, size=(700, 1))
    
    # Analysis
    result = analyze_volatility(data_array)

    print("'std_of_rolling_std':", result['std_of_rolling_std'])
    print("'std_last_m_values':", result['std_last_m_values'])
    print("'quartile_1':", result['quartile_1'])
    print("'quartile_3':", result['quartile_3'])
    print("'IQR':", result['IQR'])
    print("'lower_bound':", result['lower_bound'])
    print("'upper_bound':", result['upper_bound'])
    print("'is_within_reasonable_range':", result['is_within_reasonable_range'])