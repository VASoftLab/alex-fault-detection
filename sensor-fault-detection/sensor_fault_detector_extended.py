import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import logging
import argparse
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sensor_fault_detector')

def read_params_from_file(param_file='detector_params.txt'):
    """Read detector parameters from a text file."""
    params = {
        'significance_level': 0.01,
        'mean_threshold_modifier': 1.0,
        'variance_threshold_modifier': 1.0,
        'iqr_factor_plus': 1.5,
        'iqr_factor_minus': 1.5
    }
    
    try:
        with open(param_file, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Parse key-value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert value to appropriate type
                    try:
                        # Try to convert to float
                        value = float(value)
                        
                        # Store in params
                        if key in params:
                            params[key] = value
                    except ValueError:
                        logger.warning(f"Could not parse value for parameter {key}: {value}")
        
        
        logger.info("")
        logger.info(f"Parameters loaded from {param_file}")
        
        for key, value in params.items():
            logger.info(f"{key}: {value}")

    except Exception as e:
        logger.warning(f"Error reading parameters from {param_file}: {str(e)}. Using defaults.")
    
    return params

#------------------------------------------------------------
# DETECTOR CLASS
#------------------------------------------------------------
class OptimizedDivergenceDetector:
    """Advanced detector for identifying when one time series diverges from others."""
    
    def __init__(
        self, 
        window_size=30, 
        min_consecutive_periods=2, 
        significance_level=None,
        enable_online_learning=False,
        model_name="default_model",
        param_file='detector_params.txt'
    ):
        # Read parameters from file
        self.params = read_params_from_file(param_file)
        
        self.window_size = window_size
        self.min_consecutive_periods = min_consecutive_periods
        # Use parameter file value if not explicitly provided
        self.significance_level = significance_level if significance_level is not None else self.params['significance_level']
        self.enable_online_learning = enable_online_learning
        self.model_name = model_name
        
        # State tracking
        self.consecutive_counters = {}
        self.historical_stats = {}
        self.series_names = []
        
        # Learning parameters - read from params file
        self.mean_threshold_modifier = self.params['mean_threshold_modifier']
        self.variance_threshold_modifier = self.params['variance_threshold_modifier']
       
        logger.info("")
        logger.info(f"Initialized detector with window_size={window_size}, significance={self.significance_level}")

    def fit(self, time_series_dict):
        """Establish baseline statistics for each time series."""
        self.series_names = list(time_series_dict.keys())
        
        for name, series in time_series_dict.items():
            # Initialize counter for consecutive divergence periods
            self.consecutive_counters[name] = {
                'mean': 0,
                'variance': 0
            }
            
            # Convert to numpy array if needed
            if isinstance(series, pd.Series):
                series = series.to_numpy()
            elif not isinstance(series, np.ndarray):
                series = np.array(series)
                
            # Calculate statistics
            self.historical_stats[name] = {}
            self.historical_stats[name]['mean'] = np.mean(series)
            self.historical_stats[name]['std'] = np.std(series)
            self.historical_stats[name]['var'] = np.var(series)
            self.historical_stats[name]['q1'] = np.percentile(series, 25)
            self.historical_stats[name]['q3'] = np.percentile(series, 75)
            self.historical_stats[name]['iqr'] = self.historical_stats[name]['q3'] - self.historical_stats[name]['q1']
        
        logger.info(f"Fit completed on {len(self.series_names)} series")
        logger.info("")
        return self
    
    def detect(self, time_series_dict, analysis_window=None, current_idx=None):
        """Detect if any series is diverging from its baseline."""
        if analysis_window is None:
            analysis_window = self.window_size
        
        # Extract windows for analysis
        current_windows = {}
        for name, series in time_series_dict.items():
            if isinstance(series, pd.Series):
                series = series.to_numpy()
            elif not isinstance(series, np.ndarray):
                series = np.array(series)
        
            # Skip empty series
            if len(series) == 0:
                logger.warning(f"Series '{name}' has no data points. Skipping.")
                continue
        
            if current_idx is None:
                current_idx = len(series) - 1
        
            # Get recent window for analysis
            start_idx = max(0, current_idx - analysis_window + 1)
            window = series[start_idx:current_idx + 1]
        
            # Skip if window is empty
            if len(window) == 0:
                logger.warning(f"Analysis window for '{name}' is empty. Skipping.")
                continue
    
            current_windows[name] = window
        
        # Detect divergences
        results = {}
        
        for name, window in current_windows.items():
            if name not in self.historical_stats:
                logger.warning(f"Series {name} not in historical stats, skipping")
                continue
                
            # Calculate current stats
            current_mean = np.mean(window)
            current_var = np.var(window)
            
            # Get historical stats
            hist_mean = self.historical_stats[name]['mean']
            hist_var = self.historical_stats[name]['var']
            hist_std = self.historical_stats[name]['std']
            
            logger.info(f"hist_mean: {hist_mean}")
            logger.info(f"hist_var:  {hist_var}")
            logger.info(f"hist_std:  {hist_std}")
            logger.info(f"len(window):  {len(window)}")
            logger.info("")
            
            # Calculate changes
            mean_change = (current_mean - hist_mean) / hist_mean * 100 if abs(hist_mean) > 1e-10 else 0
            
            # Add a check for window length to avoid division by zero
            #if len(window) > 0 and hist_std > 1e-10:
            #    mean_zscore = (current_mean - hist_mean) / (hist_std / np.sqrt(len(window)))
            #else:
            #    mean_zscore = 0
                
            # Calculate changes
            if abs(hist_mean) > 1e-10:
                mean_change = (current_mean - hist_mean) / hist_mean * 100
            else:
                mean_change = 0
    
            if len(window) > 0 and hist_std > 1e-10:
                mean_zscore = (current_mean - hist_mean) / (hist_std / np.sqrt(len(window)))
            else:
                mean_zscore = 0
    
            if hist_var > 1e-10:
                variance_ratio = current_var / hist_var
            else:
                variance_ratio = 1.0
            
            #variance_ratio = current_var / hist_var if hist_var > 1e-10 else 1.0
            
            # Determine divergence
            mean_diverging = abs(mean_zscore) > (stats.norm.ppf(1 - self.significance_level/2) * self.mean_threshold_modifier)
            variance_diverging = variance_ratio > 2.0 * self.variance_threshold_modifier or variance_ratio < 0.5 / self.variance_threshold_modifier
            
            # Update consecutive counters
            if mean_diverging:
                self.consecutive_counters[name]['mean'] += 1
            else:
                self.consecutive_counters[name]['mean'] = 0
                
            if variance_diverging:
                self.consecutive_counters[name]['variance'] += 1
            else:
                self.consecutive_counters[name]['variance'] = 0
            
            # Check if persistent divergence
            persistent_mean = self.consecutive_counters[name]['mean'] >= self.min_consecutive_periods
            persistent_variance = self.consecutive_counters[name]['variance'] >= self.min_consecutive_periods
            
            # Store results
            results[name] = {
                'mean_diverging': persistent_mean,
                'variance_diverging': persistent_variance,
                'any_divergence': persistent_mean or persistent_variance,
                'mean_change': mean_change,
                'mean_zscore': mean_zscore,
                'variance_ratio': variance_ratio,
                'variance_decreased': variance_ratio < 1.0
            }
        
        return results

#------------------------------------------------------------
# UTILITY FUNCTIONS
#------------------------------------------------------------
def load_time_series_from_csv(file_path, delimiter=',', date_column=None):
    """Load time series data from a CSV file."""
    try:
        # Read CSV file
        logger.info(f"Loading time series from CSV file: {file_path}")
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        # Check if data was loaded
        if df.empty:
            logger.warning(f"CSV file {file_path} contains no data")
            return {}
        
        # Convert to dictionary of time series
        time_series_dict = {}
        
        # Set index to date column if specified
        if date_column and date_column in df.columns:
            try:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
                logger.info(f"Set {date_column} as datetime index")
            except Exception as e:
                logger.warning(f"Could not convert {date_column} to datetime: {str(e)}")
        
        # Process each column
        for column in df.columns:
            # Skip the date column if it wasn't set as index
            if column == date_column:
                continue
                
            # Clean column data
            series = df[column].copy()
            
            # Check if series contains usable numeric data
            if pd.api.types.is_numeric_dtype(series):
                # Drop NA values
                series = series.dropna()
                
                # Check if we have enough data
                if len(series) > 1:
                    time_series_dict[column] = series.values
                else:
                    logger.warning(f"Column '{column}' has insufficient data points (less than 2)")
            else:
                logger.warning(f"Column '{column}' contains non-numeric data")
                
        if not time_series_dict:
            logger.warning("No valid numeric time series found in CSV file")
            
        logger.info(f"Successfully loaded {len(time_series_dict)} time series from CSV file")
        return time_series_dict
        
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {str(e)}")
        return {}

def preprocess_time_series(series_dict, options=None):
    """Preprocess time series data for divergence detection."""
    try:
        from scipy import signal
        
        options = options or {}
        processed = {}
        
        for name, series in series_dict.items():
            try:
                # Convert to numpy array if needed
                if isinstance(series, pd.Series):
                    series = series.values
                elif not isinstance(series, np.ndarray):
                    series = np.array(series)
                    
                # Remove NaN values
                nan_mask = np.isnan(series)
                if np.any(nan_mask):
                    # Interpolate missing values
                    if options.get('interpolate_missing', False):
                        # Create interpolation function
                        valid_indices = np.where(~nan_mask)[0]
                        if len(valid_indices) < 2:
                            logger.warning(f"Series '{name}' has insufficient valid data for interpolation")
                            continue
                            
                        valid_values = series[valid_indices]
                        
                        from scipy.interpolate import interp1d
                        f = interp1d(valid_indices, valid_values, kind='linear', 
                                     bounds_error=False, fill_value="extrapolate")
                        
                        # Apply interpolation
                        all_indices = np.arange(len(series))
                        series = f(all_indices)
                    else:
                        # Just remove NaNs
                        series = series[~nan_mask]
                
                # Outlier removal using IQR method with separate factors
                if options.get('remove_outliers', False):
                    q1 = np.percentile(series, 25)
                    q3 = np.percentile(series, 75)
                    iqr = q3 - q1
                    
                    # Use separate factors for upper and lower bounds
                    iqr_factor_minus = options.get('iqr_factor_minus', 1.5)
                    iqr_factor_plus = options.get('iqr_factor_plus', 1.5)
                    
                    lower_bound = q1 - iqr_factor_minus * iqr
                    upper_bound = q3 + iqr_factor_plus * iqr
                    
                    outlier_mask = (series < lower_bound) | (series > upper_bound)
                    
                    if np.any(outlier_mask):
                        # Remove outliers
                        series = series[~outlier_mask]
                
                if len(series) < 2:
                    logger.warning(f"Series '{name}' has insufficient data points after preprocessing. Skipping.")
                    continue
                
                processed[name] = series
                
            except Exception as e:
                logger.warning(f"Error preprocessing series '{name}': {str(e)}")
                
        return processed
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return {}

def export_results(results, format='csv', output_file=None):
    """Export detection results to file."""
    # Convert results to DataFrame
    data = []
    for name, result in results.items():
        row = {
            'series_name': name,
            'diverging': result.get('any_divergence', False),
            'mean_diverging': result.get('mean_diverging', False),
            'variance_diverging': result.get('variance_diverging', False),
            'mean_change': result.get('mean_change', 0),
            'variance_ratio': result.get('variance_ratio', 1.0)
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Export to specified format
    if output_file:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(output_file))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if format.lower() == 'csv':
            df.to_csv(output_file, index=False)
        elif format.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'results': results
                }, f, indent=2, default=str)
        
        return output_file
    else:
        return df

#------------------------------------------------------------
# MAIN FUNCTION
#------------------------------------------------------------
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sensor Fault Detection')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='./data/output',
                        help='Path to output directory')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], default='csv',
                        help='Output format (default: csv)')
    parser.add_argument('--window-size', type=int, default=30,
                        help='Analysis window size')
    parser.add_argument('--min-periods', type=int, default=2,
                        help='Minimum consecutive periods for confirming divergence')
    parser.add_argument('--significance', type=float,
                        help='Statistical significance level')
    parser.add_argument('--date-column', type=str,
                        help='Name of date/time column in CSV')
    parser.add_argument('--delimiter', type=str, default=',',
                        help='CSV delimiter character')
    parser.add_argument('--preprocess', action='store_true',
                        help='Apply preprocessing to the input data')
    parser.add_argument('--remove-outliers', action='store_true',
                        help='Remove outliers during preprocessing')
    parser.add_argument('--interpolate-missing', action='store_true',
                        help='Interpolate missing values')
    parser.add_argument('--param-file', type=str, default='detector_params.txt',
                        help='Path to parameter file')
    
    args = parser.parse_args()
    
    try:
        # Load data
        time_series_dict = load_time_series_from_csv(
            args.input, 
            delimiter=args.delimiter,
            date_column=args.date_column
        )
        
        if not time_series_dict:
            logger.error("No valid time series found in input file")
            return 1
            
        logger.info(f"Loaded {len(time_series_dict)} time series from {args.input}")
        
        # Read parameters
        params = read_params_from_file(args.param_file)
        
        # Preprocess data if requested
        if args.preprocess:
            preprocess_options = {
                "interpolate_missing": args.interpolate_missing,
                "remove_outliers": args.remove_outliers,
                "iqr_factor_minus": params['iqr_factor_minus'],
                "iqr_factor_plus": params['iqr_factor_plus']
            }
            
            time_series_dict = preprocess_time_series(time_series_dict, preprocess_options)
            logger.info("Applied preprocessing to time series data")
        
        # Create detector
        detector = OptimizedDivergenceDetector(
            window_size=args.window_size,
            min_consecutive_periods=args.min_periods,
            significance_level=args.significance,
            param_file=args.param_file
        )
        
        # Fit detector
        detector.fit(time_series_dict)
        
        # Detect divergences
        results = detector.detect(time_series_dict)
        
        # Count divergences
        mean_divergences = sum(1 for r in results.values() if r.get('mean_diverging', False))
        var_divergences = sum(1 for r in results.values() if r.get('variance_diverging', False))
        any_divergences = sum(1 for r in results.values() if r.get('any_divergence', False))
        
        # Create output directory
        os.makedirs(args.output, exist_ok=True)
        
        # Export results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"results_{timestamp}.{args.format}"
        output_path = os.path.join(args.output, output_filename)
        
        export_results(results, format=args.format, output_file=output_path)
        logger.info(f"Results exported to {output_path}")
        
        # Print summary
        print("\nDivergence Detection Results:")
        print(f"- Total series analyzed: {len(results)}")
        print(f"- Series with divergences: {any_divergences}")
        print(f"- Mean divergences: {mean_divergences}")
        print(f"- Variance divergences: {var_divergences}")
        print(f"\nResults exported to: {output_path}")
        
        if any_divergences > 0:
            print("\nDivergent Series:")
            for name, result in results.items():
                if result.get('any_divergence', False):
                    print(f"- {name}")
                    if result.get('mean_diverging', False):
                        print(f"  * Mean change: {result.get('mean_change', 0):.2f}%")
                    if result.get('variance_diverging', False):
                        print(f"  * Variance {'decreased' if result.get('variance_decreased', False) else 'increased'} "
                             f"(ratio: {result.get('variance_ratio', 1.0):.2f})")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())