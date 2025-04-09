"""
Time Series Divergence Detection System

This module implements a complete system for detecting when one time series
diverges from others due to either mean changes or variance/standard deviation changes.
"""

# Import required libraries
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import os
import logging
from typing import Dict, List, Tuple, Union, Optional
import csv
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('divergence_detector')

class OptimizedDivergenceDetector:
    """
    Advanced detector for identifying when one time series diverges from others
    in terms of either mean or variance/standard deviation.
    
    This detector uses multiple statistical methods with weighted consensus and
    cross-series comparison to robustly identify divergence while minimizing
    false positives.
    """
    
    def __init__(
        self, 
        window_size: int = 30, 
        min_consecutive_periods: int = 2, 
        significance_level: float = 0.01,
        enable_online_learning: bool = False,
        model_name: str = "default_model"
    ):
        """
        Initialize the divergence detector with configurable parameters.
        
        Parameters:
        -----------
        window_size : int
            Size of analysis window (30+ recommended for stable variance estimation)
        min_consecutive_periods : int
            Number of consecutive periods required to confirm divergence
        significance_level : float
            Statistical significance threshold for tests (0.01 = 99% confidence)
        enable_online_learning : bool
            Whether to enable automatic threshold adjustment based on performance
        model_name : str
            Name for this model instance (used for saving/loading)
        """
        self.window_size = window_size
        self.min_consecutive_periods = min_consecutive_periods
        self.significance_level = significance_level
        self.enable_online_learning = enable_online_learning
        self.model_name = model_name
        
        # State tracking
        self.consecutive_counters = {}
        self.historical_stats = {}
        self.series_names = []
        self.historical_divergences = []
        
        # Performance metrics for online learning
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        
        # Learning parameters
        self.mean_threshold_modifier = 1.0
        self.variance_threshold_modifier = 1.0
        self.learning_rate = 0.05
        
        logger.info(f"Initialized {self.__class__.__name__} with window_size={window_size}, "
                   f"min_consecutive_periods={min_consecutive_periods}, "
                   f"significance_level={significance_level}")
    
    def fit(self, time_series_dict: Dict[str, Union[List, np.ndarray, pd.Series]], 
            training_period: Optional[Tuple[int, int]] = None) -> 'OptimizedDivergenceDetector':
        """
        Establish baseline statistics for each time series.
        
        Parameters:
        -----------
        time_series_dict : dict
            Dictionary mapping series names to time series data
        training_period : tuple or None
            Optional (start_idx, end_idx) to specify training data range
            
        Returns:
        --------
        self : OptimizedDivergenceDetector
            The fitted detector
        """
        self.series_names = list(time_series_dict.keys())
        
        for name, series in time_series_dict.items():
            # Initialize counter for consecutive divergence periods
            self.consecutive_counters[name] = {
                'mean': 0,
                'variance': 0
            }
            
            # Convert to numpy array if it's not already
            if isinstance(series, pd.Series):
                series = series.to_numpy()
            elif not isinstance(series, np.ndarray):
                series = np.array(series)
            
            # Use specified training period or full series
            if training_period:
                start_idx, end_idx = training_period
                training_data = series[start_idx:end_idx]
            else:
                training_data = series
                
            # Calculate and store historical statistics
            self.historical_stats[name] = {}
            
            # Basic statistics
            self.historical_stats[name]['mean'] = np.mean(training_data)
            self.historical_stats[name]['std'] = np.std(training_data)
            self.historical_stats[name]['var'] = np.var(training_data)
            self.historical_stats[name]['median'] = np.median(training_data)
            
            # Store quartiles for robust variance estimation
            self.historical_stats[name]['q1'] = np.percentile(training_data, 25)
            self.historical_stats[name]['q3'] = np.percentile(training_data, 75)
            self.historical_stats[name]['iqr'] = self.historical_stats[name]['q3'] - self.historical_stats[name]['q1']
            
            # Store time pattern information for 24-hour seasonality (if enough data)
            if len(training_data) >= 96:  # At least one day of data (assuming 15-min intervals)
                hourly_patterns = {}
                for hour in range(24):
                    # Extract data points for this hour (assuming 4 points per hour for 15-min data)
                    hour_idxs = [i for i in range(len(training_data)) if (i // 4) % 24 == hour]
                    if hour_idxs:
                        hour_data = training_data[hour_idxs]
                        hourly_patterns[hour] = {
                            'mean': np.mean(hour_data),
                            'std': np.std(hour_data)
                        }
                self.historical_stats[name]['hourly_patterns'] = hourly_patterns
            
            # Calculate cross-correlations with other series
            self.historical_stats[name]['correlations'] = {}
            for other_name, other_series in time_series_dict.items():
                if other_name != name:
                    if isinstance(other_series, pd.Series):
                        other_data = other_series.to_numpy()
                    elif not isinstance(other_series, np.ndarray):
                        other_data = np.array(other_series)
                    else:
                        other_data = other_series
                    
                    if training_period:
                        other_data = other_data[start_idx:end_idx]
                    
                    # Ensure same length for correlation
                    min_len = min(len(training_data), len(other_data))
                    corr = np.corrcoef(training_data[:min_len], other_data[:min_len])[0, 1]
                    self.historical_stats[name]['correlations'][other_name] = corr
        
        logger.info(f"Fit completed on {len(self.series_names)} series with {len(training_data)} points")
        return self
    
    def detect(self, time_series_dict: Dict[str, Union[List, np.ndarray, pd.Series]], 
               analysis_window: Optional[int] = None, 
               current_idx: Optional[int] = None) -> Dict[str, Dict]:
        """
        Detect if any series is diverging due to mean or variance changes.
        
        Parameters:
        -----------
        time_series_dict : dict
            Dictionary mapping series names to data
        analysis_window : int or None
            Number of recent points to analyze for divergence (uses default if None)
        current_idx : int or None
            Optional index to specify current time (end of analysis window)
            
        Returns:
        --------
        dict
            Results of divergence detection for each series
        """
        # Use default window if not specified
        if analysis_window is None:
            analysis_window = self.window_size
        
        # Extract windows for analysis
        current_windows = {}
        for name, series in time_series_dict.items():
            if isinstance(series, pd.Series):
                series = series.to_numpy()
            elif not isinstance(series, np.ndarray):
                series = np.array(series)
                
            if current_idx is None:
                current_idx = len(series) - 1
                
            # Get recent window for analysis
            start_idx = max(0, current_idx - analysis_window + 1)
            current_windows[name] = series[start_idx:current_idx + 1]
        
        # Get detection results for both types of divergence
        mean_results = self._detect_mean_changes(current_windows)
        variance_results = self._detect_variance_changes(current_windows)
        
        # Combine and process results
        results = self._process_results(mean_results, variance_results)
        
        # Store divergences for learning and analysis
        timestamp = datetime.now()
        for name, result in results.items():
            if result['any_divergence']:
                self.historical_divergences.append({
                    'timestamp': timestamp,
                    'series_name': name,
                    'mean_diverging': result['mean_diverging'],
                    'variance_diverging': result['variance_diverging'],
                    'mean_change': result['mean_change'],
                    'variance_ratio': result['variance_ratio'],
                    'mean_zscore': result['mean_zscore'],
                    'variance_zscore': result['f_zscore']
                })
        
        logger.info(f"Detection completed: {sum(1 for r in results.values() if r['any_divergence'])} "
                   f"series flagged out of {len(results)}")
        return results
    
    def _detect_mean_changes(self, current_windows: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Detect mean changes using multiple methods with weighted scoring.
        
        Parameters:
        -----------
        current_windows : dict
            Dictionary of current analysis windows
            
        Returns:
        --------
        dict
            Detection results for mean changes
        """
        results = {}
        
        # METHOD 1: Statistical significance test (25% weight)
        statistical_results = self._statistical_mean_test(current_windows)
        
        # METHOD 2: Cross-series relative change analysis (50% weight)
        relative_change_results = self._relative_mean_change(current_windows)
        
        # METHOD 3: Pattern consistency analysis (25% weight)
        pattern_results = self._pattern_consistency(current_windows)
        
        # Combine results with weighted scoring
        for name in self.series_names:
            # Calculate weighted score (max 4.5)
            methods_score = 0
            
            # Statistical significance (25% weight + bonus)
            stat_threshold = stats.norm.ppf(1 - self.significance_level/2) * self.mean_threshold_modifier
            stat_significant = abs(statistical_results[name]['zscore']) > stat_threshold
            
            if stat_significant:
                methods_score += 1.0
                # Bonus for very strong signal
                if abs(statistical_results[name]['zscore']) > 2 * stat_threshold:
                    methods_score += 0.5
            
            # Relative change significance (50% weight)
            rel_threshold = 2.0 * self.mean_threshold_modifier
            rel_significant = abs(relative_change_results[name]['zscore']) > rel_threshold
            
            if rel_significant:
                methods_score += 2.0
            
            # Pattern consistency (25% weight)
            if pattern_results[name]['significant']:
                methods_score += 1.0
            
            # Determine divergence (threshold at >50% of max score)
            is_diverging = methods_score >= 2.5
            
            # Update consecutive counter
            if is_diverging:
                self.consecutive_counters[name]['mean'] += 1
            else:
                self.consecutive_counters[name]['mean'] = 0
            
            # Determine if persistent divergence
            persistent_divergence = self.consecutive_counters[name]['mean'] >= self.min_consecutive_periods
            
            # Store results
            results[name] = {
                'percent_change': relative_change_results[name]['percent_change'],
                'statistical_zscore': statistical_results[name]['zscore'],
                'relative_zscore': relative_change_results[name]['zscore'],
                'pattern_consistency': pattern_results[name]['consistency'],
                'methods_score': methods_score,
                'consecutive_periods': self.consecutive_counters[name]['mean'],
                'diverging': persistent_divergence
            }
        
        return results
    
    def _detect_variance_changes(self, current_windows: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Detect variance changes using multiple methods with weighted scoring.
        
        Parameters:
        -----------
        current_windows : dict
            Dictionary of current analysis windows
            
        Returns:
        --------
        dict
            Detection results for variance changes
        """
        results = {}
        
        # METHOD 1: F-test for variance comparison (50% weight)
        f_test_results = self._f_test(current_windows)
        
        # METHOD 2: Variance ratio analysis (25% weight)
        variance_ratio_results = self._variance_ratio(current_windows)
        
        # METHOD 3: Robust IQR analysis (25% weight)
        iqr_results = self._iqr_analysis(current_windows)
        
        # Combine results with weighted scoring
        for name in self.series_names:
            # Calculate weighted score (max 4.0)
            methods_score = 0
            
            # F-test (50% weight + bonus)
            f_threshold = 2.5 * self.variance_threshold_modifier
            rel_f_threshold = 1.5 * self.variance_threshold_modifier
            
            f_significant = (f_test_results[name]['f_stat'] > f_threshold and 
                            abs(f_test_results[name]['zscore']) > rel_f_threshold)
            
            if f_significant:
                methods_score += 2.0
                # Bonus for very strong signal
                if f_test_results[name]['f_stat'] > 2 * f_threshold:
                    methods_score += 0.5
            
            # Variance ratio (25% weight)
            if variance_ratio_results[name]['significant']:
                methods_score += 1.0
            
            # IQR analysis (25% weight)
            if iqr_results[name]['significant']:
                methods_score += 1.0
            
            # Determine divergence (threshold at >60% of max score)
            is_diverging = methods_score >= 2.5
            
            # Update consecutive counter
            if is_diverging:
                self.consecutive_counters[name]['variance'] += 1
            else:
                self.consecutive_counters[name]['variance'] = 0
            
            # Determine if persistent divergence
            persistent_divergence = self.consecutive_counters[name]['variance'] >= self.min_consecutive_periods
            
            # Store results
            results[name] = {
                'variance_ratio': variance_ratio_results[name]['ratio'],
                'f_stat': f_test_results[name]['f_stat'],
                'f_zscore': f_test_results[name]['zscore'],
                'iqr_change': iqr_results[name]['change'],
                'variance_decreased': f_test_results[name]['variance_decreased'],
                'methods_score': methods_score,
                'consecutive_periods': self.consecutive_counters[name]['variance'],
                'diverging': persistent_divergence
            }
        
        return results
    
    def _statistical_mean_test(self, current_windows: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Perform statistical test for mean changes.
        """
        results = {}
        
        for name, window in current_windows.items():
            # Calculate current mean
            current_mean = np.mean(window)
            
            # Get historical statistics
            hist_mean = self.historical_stats[name]['mean']
            hist_std = self.historical_stats[name]['std']
            
            # Calculate z-score (t-statistic)
            zscore = (current_mean - hist_mean) / (hist_std / np.sqrt(len(window)))
            
            # Determine significance with appropriate threshold
            critical_value = stats.norm.ppf(1 - self.significance_level/2)
            is_significant = abs(zscore) > critical_value
            
            results[name] = {
                'zscore': zscore,
                'significant': is_significant
            }
        
        return results
    
    def _relative_mean_change(self, current_windows: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Compare relative mean changes across series.
        """
        results = {}
        
        # Calculate percent changes for all series
        percent_changes = {}
        for name, window in current_windows.items():
            current_mean = np.mean(window)
            hist_mean = self.historical_stats[name]['mean']
            
            if hist_mean != 0:
                change = (current_mean - hist_mean) / hist_mean * 100
            else:
                change = 0  # Avoid division by zero
                
            percent_changes[name] = change
        
        # Calculate z-scores of percent changes
        change_values = np.array(list(percent_changes.values()))
        mean_change = np.mean(change_values)
        std_change = np.std(change_values)
        
        for name in self.series_names:
            if std_change > 0:
                zscore = (percent_changes[name] - mean_change) / std_change
            else:
                zscore = 0
            
            # Determine significance - threshold adjusted by modifier
            threshold = 2.0 * self.mean_threshold_modifier
            is_significant = abs(zscore) > threshold
            
            results[name] = {
                'percent_change': percent_changes[name],
                'zscore': zscore,
                'significant': is_significant
            }
        
        return results
    
    def _pattern_consistency(self, current_windows: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Analyze pattern consistency to validate mean changes.
        """
        results = {}
        
        for name, window in current_windows.items():
            hist_mean = self.historical_stats[name]['mean']
            current_mean = np.mean(window)
            
            # Direction of change
            direction = 1 if current_mean > hist_mean else -1
            
            # Count points consistent with direction
            consistent_points = sum(1 for point in window if (point - hist_mean) * direction > 0)
            consistency = consistent_points / len(window)
            
            # Significant if highly consistent pattern (threshold 70%)
            is_significant = consistency > 0.7
            
            results[name] = {
                'consistency': consistency,
                'significant': is_significant
            }
        
        return results
    
    def _f_test(self, current_windows: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Perform F-test to detect variance changes.
        """
        results = {}
        
        # Calculate F-statistics for all series
        f_stats = {}
        variance_decreased = {}
        
        for name, window in current_windows.items():
            # Calculate current variance
            current_var = np.var(window)
            
            # Get historical variance
            hist_var = self.historical_stats[name]['var']
            
            # F-statistic (ratio of variances)
            # Ensure larger variance is in numerator for consistent interpretation
            if hist_var >= current_var:
                f_stat = hist_var / current_var
                variance_decreased[name] = True
            else:
                f_stat = current_var / hist_var
                variance_decreased[name] = False
            
            f_stats[name] = f_stat
        
        # Calculate z-scores of F-statistics
        f_values = np.array(list(f_stats.values()))
        mean_f = np.mean(f_values)
        std_f = np.std(f_values)
        
        for name in self.series_names:
            if std_f > 0:
                zscore = (f_stats[name] - mean_f) / std_f
            else:
                zscore = 0
            
            # Two-part significance test with adjusted thresholds:
            # 1. F-statistic exceeds critical value (absolute significance)
            # 2. F-statistic is an outlier compared to other series (relative significance)
            critical_f = 2.5 * self.variance_threshold_modifier
            rel_threshold = 1.5 * self.variance_threshold_modifier
            
            is_significant = (f_stats[name] > critical_f and abs(zscore) > rel_threshold)
            
            results[name] = {
                'f_stat': f_stats[name],
                'zscore': zscore,
                'variance_decreased': variance_decreased[name],
                'significant': is_significant
            }
        
        return results
    
    def _variance_ratio(self, current_windows: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Analyze variance ratios across series.
        """
        results = {}
        
        # Calculate variance ratios
        ratios = {}
        for name, window in current_windows.items():
            current_var = np.var(window)
            hist_var = self.historical_stats[name]['var']
            
            if hist_var > 0:
                ratio = current_var / hist_var
            else:
                ratio = 1.0  # Default if historical variance is zero
                
            ratios[name] = ratio
        
        # Calculate z-scores of ratios
        ratio_values = np.array(list(ratios.values()))
        mean_ratio = np.mean(ratio_values)
        std_ratio = np.std(ratio_values)
        
        for name in self.series_names:
            if std_ratio > 0:
                zscore = (ratios[name] - mean_ratio) / std_ratio
            else:
                zscore = 0
            
            # Significant if z-score exceeds threshold (adjusted by modifier)
            threshold = 2.0 * self.variance_threshold_modifier
            is_significant = abs(zscore) > threshold
            
            results[name] = {
                'ratio': ratios[name],
                'zscore': zscore,
                'significant': is_significant
            }
        
        return results
    
    def _iqr_analysis(self, current_windows: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Use IQR for robust variance change detection.
        """
        results = {}
        
        # Calculate IQR changes
        iqr_changes = {}
        for name, window in current_windows.items():
            # Calculate current IQR
            q1 = np.percentile(window, 25)
            q3 = np.percentile(window, 75)
            iqr = q3 - q1
            
            # Historical IQR
            hist_iqr = self.historical_stats[name]['iqr']
            
            # Calculate relative change
            if hist_iqr > 0:
                change = (iqr - hist_iqr) / hist_iqr
            else:
                change = 0
                
            iqr_changes[name] = change
        
        # Calculate z-scores of IQR changes
        change_values = np.array(list(iqr_changes.values()))
        mean_change = np.mean(change_values)
        std_change = np.std(change_values)
        
        for name in self.series_names:
            if std_change > 0:
                zscore = (iqr_changes[name] - mean_change) / std_change
            else:
                zscore = 0
            
            # Significant if z-score exceeds threshold (adjusted by modifier)
            threshold = 2.0 * self.variance_threshold_modifier
            is_significant = abs(zscore) > threshold
            
            results[name] = {
                'change': iqr_changes[name],
                'zscore': zscore,
                'significant': is_significant
            }
        
        return results
    
    def _process_results(self, mean_results: Dict[str, Dict], 
                         variance_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Process and combine results from mean and variance detection.
        
        This function:
        1. Combines mean and variance detection results
        2. Applies system-wide change filtering
        3. Handles interactions between mean and variance detection
        """
        combined_results = {}
        
        # First, combine results
        for name in self.series_names:
            combined_results[name] = {
                # Mean change metrics
                'mean_change': mean_results[name]['percent_change'],
                'mean_zscore': mean_results[name]['statistical_zscore'],
                'mean_relative_zscore': mean_results[name]['relative_zscore'],
                'mean_pattern_consistency': mean_results[name]['pattern_consistency'],
                'mean_methods_score': mean_results[name]['methods_score'],
                'mean_consecutive_periods': mean_results[name]['consecutive_periods'],
                'mean_diverging': mean_results[name]['diverging'],
                
                # Variance change metrics
                'variance_ratio': variance_results[name]['variance_ratio'],
                'f_stat': variance_results[name]['f_stat'],
                'f_zscore': variance_results[name]['f_zscore'],
                'iqr_change': variance_results[name]['iqr_change'],
                'variance_decreased': variance_results[name]['variance_decreased'],
                'variance_methods_score': variance_results[name]['methods_score'],
                'variance_consecutive_periods': variance_results[name]['consecutive_periods'],
                'variance_diverging': variance_results[name]['diverging'],
                
                # Combined status
                'any_divergence': mean_results[name]['diverging'] or variance_results[name]['diverging']
            }
        
        # Filter system-wide mean changes
        self._filter_system_wide_changes(combined_results, 'mean')
        
        # Filter system-wide variance changes
        self._filter_system_wide_changes(combined_results, 'variance')
        
        # Update 'any_divergence' flag
        for name in self.series_names:
            combined_results[name]['any_divergence'] = (
                combined_results[name]['mean_diverging'] or 
                combined_results[name]['variance_diverging']
            )
        
        return combined_results
    
    def _filter_system_wide_changes(self, results: Dict[str, Dict], change_type: str):
        """
        Filter out system-wide changes to reduce false positives.
        
        Parameters:
        -----------
        results : dict
            Detection results to filter
        change_type : str
            Type of change to filter ('mean' or 'variance')
        """
        # Get series flagged as diverging
        flagged_series = [
            name for name in self.series_names 
            if results[name][f'{change_type}_diverging']
        ]
        
        if len(flagged_series) <= 1:
            # Zero or one series flagged - no filtering needed
            return
            
        # Multiple series flagged - potential system-wide change
        if change_type == 'mean':
            # Get relative z-scores and method scores
            zscores = {name: abs(results[name]['mean_relative_zscore']) for name in flagged_series}
            scores = {name: results[name]['mean_methods_score'] for name in flagged_series}
        else:  # variance
            # Get F-statistics and method scores
            zscores = {name: abs(results[name]['f_zscore']) for name in flagged_series}
            scores = {name: results[name]['variance_methods_score'] for name in flagged_series}
        
        # Find the strongest signal
        strongest_series = max(scores.items(), key=lambda x: x[1])[0]
        max_score = scores[strongest_series]
        
        # Find second strongest
        other_scores = {k: v for k, v in scores.items() if k != strongest_series}
        second_strongest = max(other_scores.values()) if other_scores else 0
        
        # Only keep the strongest if it's significantly stronger
        if max_score > second_strongest * 1.3:
            # Reset flags for all except the strongest
            for name in flagged_series:
                if name != strongest_series:
                    results[name][f'{change_type}_diverging'] = False
        else:
            # System-wide change - reset all flags
            for name in flagged_series:
                results[name][f'{change_type}_diverging'] = False
                results[name]['system_wide_change'] = True
    
    def provide_feedback(self, feedback: Dict[str, List[str]]):
        """
        Provide feedback to the detector for online learning.
        
        Parameters:
        -----------
        feedback : dict
            Dictionary with 'true_positives', 'false_positives', and 'false_negatives' lists
            containing series names
        """
        if not self.enable_online_learning:
            logger.info("Online learning disabled - feedback ignored")
            return
        
        # Update counters
        self.true_positives += len(feedback.get('true_positives', []))
        self.false_positives += len(feedback.get('false_positives', []))
        self.false_negatives += len(feedback.get('false_negatives', []))
        
        # Calculate precision and recall
        if self.true_positives + self.false_positives > 0:
            precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            precision = 1.0
            
        if self.true_positives + self.false_negatives > 0:
            recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            recall = 1.0
        
        # Adjust thresholds based on feedback
        if self.false_positives > self.false_negatives:
            # Too many false positives - make thresholds more conservative
            self.mean_threshold_modifier *= (1 + self.learning_rate)
            self.variance_threshold_modifier *= (1 + self.learning_rate)
            
            logger.info(f"Adjusted thresholds to reduce false positives: "
                       f"mean_modifier={self.mean_threshold_modifier:.3f}, "
                       f"variance_modifier={self.variance_threshold_modifier:.3f}")
        
        elif self.false_negatives > self.false_positives:
            # Too many false negatives - make thresholds more sensitive
            self.mean_threshold_modifier *= (1 - self.learning_rate)
            self.variance_threshold_modifier *= (1 - self.learning_rate)
            
            logger.info(f"Adjusted thresholds to reduce false negatives: "
                       f"mean_modifier={self.mean_threshold_modifier:.3f}, "
                       f"variance_modifier={self.variance_threshold_modifier:.3f}")

    def explain_divergence(self, time_series_dict: Dict[str, Union[List, np.ndarray, pd.Series]], 
                          series_name: str, results: Dict[str, Dict]) -> str:
        """
        Provide a detailed explanation of why a series was flagged as diverging.
        
        Parameters:
        -----------
        time_series_dict : dict
            Dictionary mapping series names to time series data
        series_name : str
            Name of the series to explain
        results : dict
            Detection results from the detect method
            
        Returns:
        --------
        str
            Human-readable explanation of the divergence
        """
        if series_name not in self.series_names:
            return f"Series '{series_name}' not found in monitored data."
        
        if not results[series_name]['any_divergence']:
            return f"Series '{series_name}' is not currently diverging from normal patterns."
        
        # Extract current window for analysis
        series = time_series_dict[series_name]
        if isinstance(series, pd.Series):
            series = series.to_numpy()
        elif not isinstance(series, np.ndarray):
            series = np.array(series)
            
        current_window = series[-self.window_size:]
        
        explanation = [f"DIVERGENCE EXPLANATION FOR '{series_name}'"]
        explanation.append("=" * 50)
        
        # Explain mean divergence if detected
        if results[series_name]['mean_diverging']:
            explanation.append("\nMEAN CHANGE DETECTED:")
            explanation.append("-" * 30)
            
            # Basic statistics
            current_mean = np.mean(current_window)
            hist_mean = self.historical_stats[series_name]['mean']
            percent_change = results[series_name]['mean_change']
            direction = "increased" if current_mean > hist_mean else "decreased"
            
            explanation.append(f"• The mean value has {direction} by {abs(percent_change):.2f}%")
            explanation.append(f"• Historical baseline mean: {hist_mean:.4f}")
            explanation.append(f"• Current mean: {current_mean:.4f}")
            
            # Statistical significance
            zscore = results[series_name]['mean_zscore']
            explanation.append(f"• Statistical significance: z-score = {zscore:.2f} " +
                              f"(values above {stats.norm.ppf(1 - self.significance_level/2):.2f} are significant)")
            
            # Cross-series comparison
            rel_zscore = results[series_name]['mean_relative_zscore']
            explanation.append(f"• Relative to other series: z-score = {rel_zscore:.2f} " +
                              "(indicates how unusual this change is compared to other series)")
            
            # Pattern consistency
            consistency = results[series_name]['mean_pattern_consistency']
            explanation.append(f"• Pattern consistency: {consistency:.2f} " +
                              f"({int(consistency * 100)}% of recent values follow the same direction)")
            
            # Duration
            consecutive = results[series_name]['mean_consecutive_periods']
            explanation.append(f"• Duration: Divergence detected for {consecutive} consecutive periods")
            
        # Explain variance divergence if detected
        if results[series_name]['variance_diverging']:
            explanation.append("\nVARIANCE CHANGE DETECTED:")
            explanation.append("-" * 30)
            
            # Basic statistics
            current_var = np.var(current_window)
            hist_var = self.historical_stats[series_name]['var']
            variance_ratio = results[series_name]['variance_ratio']
            
            if results[series_name]['variance_decreased']:
                explanation.append(f"• The variance has decreased by {(1 - variance_ratio) * 100:.2f}%")
            else:
                explanation.append(f"• The variance has increased by {(variance_ratio - 1) * 100:.2f}%")
            
            explanation.append(f"• Historical baseline variance: {hist_var:.4f}")
            explanation.append(f"• Current variance: {current_var:.4f}")
            
            # F-statistic
            f_stat = results[series_name]['f_stat']
            explanation.append(f"• F-statistic: {f_stat:.2f} " +
                              f"(values above {2.5 * self.variance_threshold_modifier:.2f} are significant)")
            
            # Cross-series comparison
            f_zscore = results[series_name]['f_zscore']
            explanation.append(f"• Relative to other series: z-score = {f_zscore:.2f} " +
                              "(indicates how unusual this variance change is compared to other series)")
            
            # Duration
            consecutive = results[series_name]['variance_consecutive_periods']
            explanation.append(f"• Duration: Divergence detected for {consecutive} consecutive periods")
        
        # Recommended actions
        explanation.append("\nRECOMMENDED ACTIONS:")
        explanation.append("-" * 30)
        
        if results[series_name]['mean_diverging']:
            explanation.append("• Investigate factors that could cause the mean level to " +
                              ("increase" if current_mean > hist_mean else "decrease"))
        
        if results[series_name]['variance_diverging']:
            if results[series_name]['variance_decreased']:
                explanation.append("• Investigate potential stabilizing factors or reduced variability")
            else:
                explanation.append("• Investigate sources of increased volatility or instability")
        
        explanation.append("• Check for changes in external factors or operating conditions")
        explanation.append("• Verify sensor calibration if applicable")
        explanation.append("• Consider updating baseline if this represents a new normal state")
        
        return "\n".join(explanation)
    
    def save_model(self, filepath: str = None):
        """
        Save the trained detector to disk.
        
        Parameters:
        -----------
        filepath : str or None
            Path to save the model (default: '{model_name}.pkl')
        """
        if filepath is None:
            filepath = f"{self.model_name}.pkl"
            
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
        logger.info(f"Model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath: str):
        """
        Load a trained detector from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        OptimizedDivergenceDetector
            The loaded detector
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Model loaded from {filepath}")
        return model


class DivergenceMonitoringSystem:
    """
    Complete monitoring system for detecting and alerting on time series divergence.
    
    This system wraps the OptimizedDivergenceDetector with additional functionality
    for alerting, visualization, and baseline management.
    """
    
    def __init__(
        self,
        window_size: int = 30,
        min_consecutive_periods: int = 2,
        significance_level: float = 0.01,
        enable_online_learning: bool = False,
        enable_alerts: bool = True,
        model_name: str = "monitoring_system",
        storage_dir: str = "./models"
    ):
        """
        Initialize the monitoring system.
        
        Parameters:
        -----------
        window_size : int
            Size of analysis window
        min_consecutive_periods : int
            Required consecutive periods before alerting
        significance_level : float
            Statistical significance threshold
        enable_online_learning : bool
            Whether to enable detector threshold adjustment
        enable_alerts : bool
            Whether to enable alerting
        model_name : str
            Name for this monitoring system
        storage_dir : str
            Directory for storing models and data
        """
        # Initialize the detector
        self.detector = OptimizedDivergenceDetector(
            window_size=window_size,
            min_consecutive_periods=min_consecutive_periods,
            significance_level=significance_level,
            enable_online_learning=enable_online_learning,
            model_name=model_name
        )
        
        self.model_name = model_name
        self.storage_dir = storage_dir
        self.enable_alerts = enable_alerts
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Alerting state
        self.alert_state = {}
        self.alert_history = []
        
        # Baseline management
        self.baseline_update_interval = 30  # days
        self.last_baseline_update = datetime.now()
        
        logger.info(f"Initialized {self.__class__.__name__} with model_name={model_name}")
    
    def fit(self, time_series_dict: Dict[str, Union[List, np.ndarray, pd.Series]], 
            training_period: Optional[Tuple[int, int]] = None) -> 'DivergenceMonitoringSystem':
        """
        Fit the detector with historical data.
        
        Parameters:
        -----------
        time_series_dict : dict
            Dictionary mapping series names to data
        training_period : tuple or None
            Optional (start_idx, end_idx) to specify training data range
            
        Returns:
        --------
        self : DivergenceMonitoringSystem
            The fitted system
        """
        # Initialize alert state for each series
        for name in time_series_dict.keys():
            self.alert_state[name] = {
                'mean_alerted': False,
                'variance_alerted': False,
                'last_mean_alert': None,
                'last_variance_alert': None
            }
        
        # Fit the detector
        self.detector.fit(time_series_dict, training_period)
        
        # Save the initial model
        self._save_state()
        
        return self
    
    def update(self, time_series_dict: Dict[str, Union[List, np.ndarray, pd.Series]], 
              analysis_window: Optional[int] = None, 
              current_idx: Optional[int] = None) -> Dict[str, Dict]:
        """
        Update the system with new data and detect divergence.
        
        Parameters:
        -----------
        time_series_dict : dict
            Dictionary mapping series names to data
        analysis_window : int or None
            Number of recent points to analyze
        current_idx : int or None
            Optional index to specify current time
            
        Returns:
        --------
        dict
            Detection results for each series
        """
        # Run detection
        results = self.detector.detect(time_series_dict, analysis_window, current_idx)
        
        # Process alerts
        self._process_alerts(results, time_series_dict)
        
        # Check if baseline update is due
        self._check_baseline_update(time_series_dict)
        
        return results
    
    def _process_alerts(self, results: Dict[str, Dict], 
                       time_series_dict: Dict[str, Union[List, np.ndarray, pd.Series]]):
        """
        Process alerts based on detection results.
        
        Parameters:
        -----------
        results : dict
            Detection results from the detector
        time_series_dict : dict
            Dictionary mapping series names to data
        """
        if not self.enable_alerts:
            return
            
        timestamp = datetime.now()
        
        for name, result in results.items():
            # Process mean divergence alerts
            if result['mean_diverging'] and not self.alert_state[name]['mean_alerted']:
                # Generate alert
                alert = {
                    'timestamp': timestamp,
                    'series_name': name,
                    'type': 'mean_change',
                    'percent_change': result['mean_change'],
                    'zscore': result['mean_zscore'],
                    'explanation': self.detector.explain_divergence(time_series_dict, name, results)
                }
                
                self.alert_history.append(alert)
                self.alert_state[name]['mean_alerted'] = True
                self.alert_state[name]['last_mean_alert'] = timestamp
                
                logger.warning(f"ALERT: Mean change detected in series '{name}'")
                
            elif not result['mean_diverging'] and self.alert_state[name]['mean_alerted']:
                # Reset alert state
                self.alert_state[name]['mean_alerted'] = False
                logger.info(f"Alert cleared: Mean of series '{name}' returned to normal")
                
            # Process variance divergence alerts
            if result['variance_diverging'] and not self.alert_state[name]['variance_alerted']:
                # Generate alert
                alert = {
                    'timestamp': timestamp,
                    'series_name': name,
                    'type': 'variance_change',
                    'variance_ratio': result['variance_ratio'],
                    'f_stat': result['f_stat'],
                    'variance_decreased': result['variance_decreased'],
                    'explanation': self.detector.explain_divergence(time_series_dict, name, results)
                }
                
                self.alert_history.append(alert)
                self.alert_state[name]['variance_alerted'] = True
                self.alert_state[name]['last_variance_alert'] = timestamp
                
                logger.warning(f"ALERT: Variance change detected in series '{name}'")
                
            elif not result['variance_diverging'] and self.alert_state[name]['variance_alerted']:
                # Reset alert state
                self.alert_state[name]['variance_alerted'] = False
                logger.info(f"Alert cleared: Variance of series '{name}' returned to normal")
    
    def _check_baseline_update(self, time_series_dict: Dict[str, Union[List, np.ndarray, pd.Series]]):
        """
        Check if baseline update is due and perform update if needed.
        
        Parameters:
        -----------
        time_series_dict : dict
            Dictionary mapping series names to current data
        """
        # Check if update interval has passed
        now = datetime.now()
        days_since_update = (now - self.last_baseline_update).days
        
        if days_since_update >= self.baseline_update_interval:
            logger.info(f"Performing scheduled baseline update after {days_since_update} days")
            
            # Identify non-diverging series to update baselines
            results = self.detector.detect(time_series_dict)
            normal_series = {
                name: series for name, series in time_series_dict.items()
                if not results[name]['any_divergence']
            }
            
            if normal_series:
                # Update baselines using recent data from normal series
                for name, series in normal_series.items():
                    if isinstance(series, pd.Series):
                        series_data = series.to_numpy()
                    elif not isinstance(series, np.ndarray):
                        series_data = np.array(series)
                    else:
                        series_data = series
                    
                    # Use recent data for update (last 30 days equivalent)
                    recent_points = min(len(series_data), 30 * 24 * 4)  # Approx 30 days of 15-min data
                    recent_data = series_data[-recent_points:]
                    
                    # Update statistics (with slight weighting to historical values)
                    hist_stats = self.detector.historical_stats[name]
                    hist_weight = 0.7  # Weight for historical values (preserves some stability)
                    
                    # Update basic statistics
                    hist_stats['mean'] = hist_weight * hist_stats['mean'] + (1 - hist_weight) * np.mean(recent_data)
                    hist_stats['std'] = hist_weight * hist_stats['std'] + (1 - hist_weight) * np.std(recent_data)
                    hist_stats['var'] = hist_weight * hist_stats['var'] + (1 - hist_weight) * np.var(recent_data)
                    hist_stats['median'] = hist_weight * hist_stats['median'] + (1 - hist_weight) * np.median(recent_data)
                    
                    # Update quartiles
                    hist_stats['q1'] = hist_weight * hist_stats['q1'] + (1 - hist_weight) * np.percentile(recent_data, 25)
                    hist_stats['q3'] = hist_weight * hist_stats['q3'] + (1 - hist_weight) * np.percentile(recent_data, 75)
                    hist_stats['iqr'] = hist_stats['q3'] - hist_stats['q1']
                
                # Log the update
                logger.info(f"Updated baselines for {len(normal_series)} series")
            else:
                logger.warning("Baseline update skipped - all series are currently diverging")
            
            # Save updated state
            self._save_state()
            
            # Update timestamp
            self.last_baseline_update = now
    
    def dashboard(self, time_series_dict: Dict[str, Union[List, np.ndarray, pd.Series]], 
                 lookback_periods: int = 96, 
                 output_file: Optional[str] = None):
        """
        Generate a visualization dashboard of current status and historical alerts.
        
        Parameters:
        -----------
        time_series_dict : dict
            Dictionary mapping series names to data
        lookback_periods : int
            Number of historical periods to include in visualization
        output_file : str or None
            Path to save the dashboard (if None, displays interactively)
        """
        # Create figure with subplots
        n_series = len(time_series_dict)
        fig, axs = plt.subplots(n_series, 2, figsize=(15, 4 * n_series))
        
        # Adjust for single series case
        if n_series == 1:
            axs = np.array([axs])
        
        # Get detection results
        results = self.detector.detect(time_series_dict)
        
        # Plot each series
        for i, (name, series) in enumerate(time_series_dict.items()):
            # Convert to numpy if needed
            if isinstance(series, pd.Series):
                series_values = series.to_numpy()
                if series.index.dtype.kind in 'mM':  # datetime index
                    x_values = series.index
                else:
                    x_values = np.arange(len(series_values))
            else:
                series_values = np.array(series)
                x_values = np.arange(len(series_values))
            
            # Get recent values
            recent_values = series_values[-lookback_periods:]
            if len(x_values) >= lookback_periods:
                recent_x = x_values[-lookback_periods:]
            else:
                recent_x = x_values
            
            # Time series plot
            axs[i, 0].plot(recent_x, recent_values, 'b-', label=name)
            
            # Add baseline mean and confidence intervals
            baseline_mean = self.detector.historical_stats[name]['mean']
            baseline_std = self.detector.historical_stats[name]['std']
            
            axs[i, 0].axhline(y=baseline_mean, color='g', linestyle='-', label='Baseline Mean')
            axs[i, 0].axhline(y=baseline_mean + 2 * baseline_std, color='r', linestyle='--', 
                             label='±2σ Confidence')
            axs[i, 0].axhline(y=baseline_mean - 2 * baseline_std, color='r', linestyle='--')
            
            # Highlight divergence if present
            if results[name]['any_divergence']:
                if results[name]['mean_diverging']:
                    axs[i, 0].set_facecolor('#ffe6e6')  # Light red background
                    axs[i, 0].set_title(f"{name} - MEAN DIVERGENCE DETECTED", color='red')
                elif results[name]['variance_diverging']:
                    axs[i, 0].set_facecolor('#e6f2ff')  # Light blue background
                    axs[i, 0].set_title(f"{name} - VARIANCE DIVERGENCE DETECTED", color='blue')
            else:
                axs[i, 0].set_title(name)
            
            axs[i, 0].legend(loc='upper right')
            axs[i, 0].grid(True)
            
            # Distribution plot (histogram)
            axs[i, 1].hist(recent_values, bins=20, alpha=0.5, label='Recent')
            
            # Create baseline distribution for comparison
            historical_mean = self.detector.historical_stats[name]['mean']
            historical_std = self.detector.historical_stats[name]['std']
            
            x = np.linspace(historical_mean - 4 * historical_std, 
                            historical_mean + 4 * historical_std, 100)
            y = stats.norm.pdf(x, historical_mean, historical_std)
            
            # Scale to match histogram
            hist_height = np.histogram(recent_values, bins=20)[0].max()
            pdf_height = y.max()
            scale_factor = hist_height / pdf_height if pdf_height > 0 else 1
            
            axs[i, 1].plot(x, y * scale_factor, 'r-', label='Baseline Distribution')
            
            # Add current mean marker
            current_mean = np.mean(recent_values)
            axs[i, 1].axvline(x=current_mean, color='blue', linestyle='-', label='Current Mean')
            
            axs[i, 1].set_title(f"{name} - Distribution")
            axs[i, 1].legend(loc='upper right')
            axs[i, 1].grid(True)
        
        # Add global title
        plt.suptitle('Time Series Divergence Monitoring Dashboard', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save or display
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Dashboard saved to {output_file}")
        else:
            plt.show()
    
    def _save_state(self):
        """Save the current state of the monitoring system to disk."""
        # Save detector
        detector_path = os.path.join(self.storage_dir, f"{self.model_name}_detector.pkl")
        self.detector.save_model(detector_path)
        
        # Save alert history and state
        state_path = os.path.join(self.storage_dir, f"{self.model_name}_state.pkl")
        state = {
            'alert_history': self.alert_history,
            'alert_state': self.alert_state,
            'last_baseline_update': self.last_baseline_update
        }
        
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
            
        logger.info(f"Monitoring system state saved to {self.storage_dir}")
    
    @classmethod
    def load_system(cls, model_name: str, storage_dir: str = "./models") -> 'DivergenceMonitoringSystem':
        """
        Load a saved monitoring system.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to load
        storage_dir : str
            Directory containing saved models
            
        Returns:
        --------
        DivergenceMonitoringSystem
            The loaded monitoring system
        """
        # Load detector
        detector_path = os.path.join(storage_dir, f"{model_name}_detector.pkl")
        detector = OptimizedDivergenceDetector.load_model(detector_path)
        
        # Create new system instance
        system = cls(
            window_size=detector.window_size,
            min_consecutive_periods=detector.min_consecutive_periods,
            significance_level=detector.significance_level,
            enable_online_learning=detector.enable_online_learning,
            model_name=model_name,
            storage_dir=storage_dir
        )
        
        # Replace detector with loaded one
        system.detector = detector
        
        # Load state
        state_path = os.path.join(storage_dir, f"{model_name}_state.pkl")
        with open(state_path, 'rb') as f:
            state = pickle.load(f)
            
        system.alert_history = state['alert_history']
        system.alert_state = state['alert_state']
        system.last_baseline_update = state['last_baseline_update']
        
        logger.info(f"Monitoring system loaded from {storage_dir}")
        return system


def read_csv_to_dict(file_path):
    data_dict = {}
    
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        for row in csv_reader:
            if len(row) >= 2:  # Проверяем, что в строке есть как минимум два столбца
                try:                    
                    datetime_obj = datetime.strptime(row[0].strip(), "%m/%d/%y %H:%M")
                    # Парсим вещественное число
                    value = float(row[1].strip())
                    data_dict[datetime_obj] = value
                except ValueError as e:
                    print(f"Ошибка при обработке строки {row}: {e}")
    
    return data_dict

def read_csv_to_dictA(file_path):
    data_dict = {}
    
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        index = 0;
        for row in csv_reader:
            if len(row) >= 2:  # Проверяем, что в строке есть как минимум два столбца
                try:                    
                    datetime_obj = datetime.strptime(row[0].strip(), "%m/%d/%y %H:%M")
                    # Парсим вещественное число
                    value = float(row[1].strip())
                    data_dict[index] = value
                    index = index + 1
                except ValueError as e:
                    print(f"Ошибка при обработке строки {row}: {e}")
    
    return data_dict


if __name__ == "__main__":
        # Initialize the monitoring system
        monitor = DivergenceMonitoringSystem(
            window_size=96, # 24 hours of 15-minute data
            min_consecutive_periods=2,
            significance_level=0.01,
            enable_alerts=True
            )
        
        time_series_dict = read_csv_to_dict("c:\\sources\\fault-detection\\data-file-timeseries.csv")
        
        # Fit with historical data
        monitor.fit(time_series_dict, training_period=(0, 128))
        # Run detection on new data
        results = monitor.update(time_series_dict)
        # Check for diverging series
        for name, result in results.items():
            if result['any_divergence']:
                print(f"{name} is diverging!")
                explanation = monitor.detector.explain_divergence(time_series_dict, name, results)
                print(explanation)
                
        # Generate visualization dashboard
        monitor.dashboard(time_series_dict)
        # Save the model for future use
        monitor._save_state()