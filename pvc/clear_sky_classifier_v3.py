import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, auc
from typing import List, Dict, Tuple, Optional, Union
from pvlib import solarposition, clearsky, atmosphere, irradiance
from pvc.clear_sky_classifier import ClearSkyClassifier, fix_clear_sky_nans_with_nearby_zenith_medians


class ClearSkyClassifierV3(ClearSkyClassifier):
    """
    Enhanced version of ClearSkyClassifier with improvements targeting moderate irradiance false positives.
    Adds enhanced temporal context analysis and improved direct-diffuse balance features.
    """
    
    def __init__(self, 
                 model_params: Optional[Dict] = None,
                 feature_engineering: bool = True,
                 handle_imbalance: bool = True,
                 threshold: float = 0.5,
                 latitude: Optional[float] = None,
                 longitude: Optional[float] = None,
                 altitude: Optional[float] = 0,
                 timezone: Optional[str] = 'UTC'):
        """
        Initialize the enhanced classifier with the same parameters as the base classifier.
        """
        # Call the parent class constructor
        super().__init__(
            model_params=model_params,
            feature_engineering=feature_engineering,
            handle_imbalance=handle_imbalance,
            threshold=threshold,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            timezone=timezone
        )

        self.more_categorical_features = []
        
    def _add_enhanced_temporal_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced temporal context features to target moderate irradiance false positives.
        
        Args:
            df: DataFrame with basic features already calculated
            
        Returns:
            DataFrame with additional temporal context features
        """
        
        # Define longer rolling windows for stability analysis
        longer_windows = [15, 20, 30]  # Minutes
        
        # Target irradiance components
        irradiance_cols = ['global_horizontal', 'direct_normal', 'diffuse_horizontal']
        
        # 1. Add longer rolling windows for each irradiance component
        for col in irradiance_cols:
            if col in df.columns:
                for window in longer_windows:
                    # Rolling mean for longer windows
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(
                        window=window, min_periods=max(3, window//5), center=True
                    ).mean()
                    
                    # Rolling std for longer windows (stability over longer periods)
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(
                        window=window, min_periods=max(3, window//5), center=True
                    ).std().fillna(0)
                    
                    # Coefficient of variation (normalized stability)
                    mean_col = df[f'{col}_rolling_mean_{window}']
                    std_col = df[f'{col}_rolling_std_{window}']
                    # Avoid division by zero
                    df[f'{col}_coef_var_{window}'] = std_col / mean_col.replace(0, np.nan).fillna(1)
                    # Clip to reasonable range and fill NaNs
                    df[f'{col}_coef_var_{window}'] = df[f'{col}_coef_var_{window}'].clip(0, 1).fillna(0)
        
        # 2. Add rate-of-change thresholds for sustained stability detection
        for col in irradiance_cols:
            if col in df.columns:
                # Calculate rolling max rate of change over different windows
                for window in [5, 10, 15]:
                    # Absolute rate of change
                    abs_roc = df[col].diff().abs()
                    # Rolling max of absolute rate of change
                    df[f'{col}_max_roc_{window}'] = abs_roc.rolling(
                        window=window, min_periods=2
                    ).max().fillna(0)
                    
                    # Calculate stability score (inverse of max rate of change)
                    max_val = df[col].max()
                    df[f'{col}_stability_score_{window}'] = 1 - (
                        df[f'{col}_max_roc_{window}'] / (max_val or 1)
                    ).clip(0, 1)
        
        # 3. Historical pattern comparison (compare with values at similar solar positions)
        if 'zenith' in df.columns and df.index.is_monotonic_increasing:
            # Bin zenith angles by 2 degrees
            df['zenith_bin'] = (df['zenith'] // 2) * 2
            self.more_categorical_features.append('zenith_bin')

            # For each irradiance component
            for col in irradiance_cols:
                if col in df.columns:
                    # Calculate historical zenith-specific statistics
                    # Group by zenith bin and calculate expanding statistics
                    df[f'{col}_zenith_mean'] = df.groupby('zenith_bin')[col].transform(
                        lambda x: x.expanding().mean().shift()
                    )
                    df[f'{col}_zenith_std'] = df.groupby('zenith_bin')[col].transform(
                        lambda x: x.expanding().std().shift()
                    ).fillna(0)
                    
                    # Calculate z-score distance from historical zenith-specific mean
                    mean = df[f'{col}_zenith_mean']
                    std = df[f'{col}_zenith_std']
                    # Use standard deviation + epsilon to avoid division by zero
                    df[f'{col}_zenith_zscore'] = (df[col] - mean) / (std + 1e-5)
                    df[f'{col}_zenith_zscore'] = df[f'{col}_zenith_zscore'].clip(-5, 5).fillna(0)
                    
                    # Create binary features for unusual deviations
                    df[f'{col}_unusual_high'] = (df[f'{col}_zenith_zscore'] > 2).astype(int)
                    df[f'{col}_unusual_low'] = (df[f'{col}_zenith_zscore'] < -2).astype(int)
        
        # 4. Calculate cross-component correlation over various time windows - FIX HERE
        # This detects if all components change together (clear sky) or independently (clouds)
        
        # for window in [10, 20, 30]:
        #     # Only proceed if we have all necessary components
        #     if all(col in df.columns for col in ['global_horizontal', 'direct_normal']):
        #         # Instead of using rolling.apply() with a correlation function (which caused the error),
        #         # We'll calculate this more efficiently and robustly
                
        #         # Initialize the correlation column
        #         df[f'ghi_dni_corr_{window}'] = 0.0
                
        #         # Skip if we don't have enough data
        #         if len(df) <= window:
        #             continue
                    
        #         # Calculate rolling correlation manually
        #         # We'll loop through the dataframe in chunks to avoid excessive computation
        #         chunk_size = 1000  # Process in chunks for efficiency
        #         for start_idx in range(0, len(df), chunk_size):
        #             end_idx = min(start_idx + chunk_size + window, len(df))
        #             chunk = df.iloc[start_idx:end_idx]
                    
        #             # Calculate correlation for each valid window in the chunk
        #             ghi_values = chunk['global_horizontal'].values
        #             dni_values = chunk['direct_normal'].values
                    
        #             for i in range(window, len(chunk)):
        #                 if i >= window:
        #                     # Extract window of values
        #                     ghi_window = ghi_values[i-window:i]
        #                     dni_window = dni_values[i-window:i]
                            
        #                     # Calculate correlation if we have valid data
        #                     ghi_std = np.std(ghi_window)
        #                     dni_std = np.std(dni_window)
                            
        #                     if ghi_std > 0 and dni_std > 0:
        #                         # Calculate Pearson correlation coefficient
        #                         correlation = np.corrcoef(ghi_window, dni_window)[0, 1]
                                
        #                         # Store in the dataframe
        #                         idx_in_df = start_idx + i
        #                         if idx_in_df < len(df):
        #                             df.iloc[idx_in_df, df.columns.get_loc(f'ghi_dni_corr_{window}')] = correlation
        
        # 5. Sustained clear sky detection
        # Detect periods where multiple indicators suggest clear sky for a sustained period
        if 'clear_sky_score' in df.columns:
            for window in [10, 20, 30]:
                # Calculate percentage of time the clear sky score is above 0.7 in the window
                df[f'sustained_clear_{window}'] = df['clear_sky_score'].rolling(
                    window=window, min_periods=window//2
                ).apply(lambda x: np.mean(x > 0.7)).fillna(0)
        
        return df

    def _add_improved_direct_diffuse_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add improved direct-diffuse balance features focusing on moderate irradiance conditions.
        
        Args:
            df: DataFrame with basic features already calculated
            
        Returns:
            DataFrame with additional direct-diffuse balance features
        """
        
        # Ensure we have the necessary columns
        required_cols = ['direct_normal', 'diffuse_horizontal', 'global_horizontal', 
                         'clear_dni', 'clear_dhi', 'clear_ghi', 'zenith']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Warning: Missing columns for direct-diffuse balance features: {missing}")
            # Return original dataframe if critical columns are missing
            if 'direct_normal' not in df.columns or 'diffuse_horizontal' not in df.columns:
                return df
        
        # 1. Theoretical vs. Actual Ratio Comparison
        # Compare measured ratios with expected theoretical ratios
        
        # Calculate theoretical direct/diffuse ratio from clear sky model
        if all(col in df.columns for col in ['clear_dni', 'clear_dhi']):
            # Theoretical ratio from clear sky model
            df['theoretical_direct_diffuse_ratio'] = df['clear_dni'] / df['clear_dhi'].replace(0, np.nan)
            df['theoretical_direct_diffuse_ratio'] = df['theoretical_direct_diffuse_ratio'].fillna(0).clip(0, 20)
            
            # Compare actual ratio to theoretical ratio
            if 'direct_to_diffuse_ratio' in df.columns:
                # Ratio of actual to theoretical ratio (1.0 = perfect match to clear sky model)
                df['direct_diffuse_ratio_match'] = (
                    df['direct_to_diffuse_ratio'] / df['theoretical_direct_diffuse_ratio'].replace(0, np.nan)
                ).fillna(0).clip(0, 2)
        
        # 2. Zenith-Normalized Diffuse Analysis
        # Create diffuse fraction features normalized by zenith angle
        if 'zenith' in df.columns:
            # Bin zenith angles by 5 degrees
            # df['zenith_bin'] = (df['zenith'] // 5) * 5
            
            # Calculate typical diffuse fraction at each zenith bin
            if all(col in df.columns for col in ['diffuse_horizontal', 'global_horizontal']):
                # Calculate diffuse fraction
                df['diffuse_fraction'] = df['diffuse_horizontal'] / df['global_horizontal'].replace(0, np.nan)
                df['diffuse_fraction'] = df['diffuse_fraction'].fillna(0).clip(0, 1)
                
                # Calculate median diffuse fraction for each zenith bin
                zenith_diffuse_median = df.groupby('zenith_bin')['diffuse_fraction'].transform('median')
                
                # Calculate deviation from typical diffuse fraction for this zenith
                df['diffuse_fraction_deviation'] = df['diffuse_fraction'] - zenith_diffuse_median
                
                # Calculate normalized diffuse fraction (adjusted for zenith angle)
                # Higher values at higher zenith angles are normal
                cos_zenith = np.cos(np.radians(df['zenith'].clip(0, 89)))
                df['zenith_adjusted_diffuse_fraction'] = df['diffuse_fraction'] * cos_zenith
        
        # 3. Direct and Diffuse Consistency Features
        # Create features that check if direct and diffuse components are consistent with each other
        
        # Consistency of direct/diffuse/global relationship
        if all(col in df.columns for col in ['direct_normal', 'diffuse_horizontal', 'global_horizontal', 'zenith']):
            # Calculate theoretical global from direct and diffuse components
            cos_zenith = np.cos(np.radians(df['zenith'].clip(0, 89)))
            theoretical_global = df['direct_normal'] * cos_zenith + df['diffuse_horizontal']
            
            # Difference between measured global and theoretical global
            df['global_consistency_error'] = np.abs(df['global_horizontal'] - theoretical_global)
            
            # Normalize by global horizontal to get relative error
            df['global_relative_consistency_error'] = (
                df['global_consistency_error'] / df['global_horizontal'].replace(0, np.nan)
            ).fillna(0).clip(0, 1)
            
            # Create binary feature for inconsistent measurements
            df['inconsistent_measurements'] = (df['global_relative_consistency_error'] > 0.15).astype(int)
        
        # 4. Component Balance Analytics
        # Analyze the balance between components for different conditions
        
        if all(col in df.columns for col in ['direct_normal', 'diffuse_horizontal', 'global_horizontal']):
            # Calculate diffuse to global ratio
            if 'diffuse_to_global_ratio' not in df.columns:
                df['diffuse_to_global_ratio'] = (
                    df['diffuse_horizontal'] / df['global_horizontal'].replace(0, np.nan)
                ).fillna(0).clip(0, 1)
            
            # Create features for different atmospheric conditions based on component balance
            conditions = [
                # Clear sky: High direct, low diffuse
                (df['direct_to_diffuse_ratio'] > 5) & (df['diffuse_to_global_ratio'] < 0.25),
                
                # Thin clouds: Moderate direct, moderate diffuse
                (df['direct_to_diffuse_ratio'] > 1) & 
                (df['direct_to_diffuse_ratio'] <= 5) & 
                (df['diffuse_to_global_ratio'] >= 0.25) & 
                (df['diffuse_to_global_ratio'] < 0.5),
                
                # Hazy/Polluted: Moderate direct, high diffuse
                (df['direct_to_diffuse_ratio'] > 1) & 
                (df['diffuse_to_global_ratio'] >= 0.5),
                
                # Cloudy: Low direct, high diffuse
                (df['direct_to_diffuse_ratio'] <= 1)
            ]
            
            choices = ['clear_sky', 'thin_clouds', 'hazy', 'cloudy']
            choices_code = [0, 1, 2, 3]
            df['component_balance_condition_code'] = np.select(conditions, choices_code, default=4)
            self.more_categorical_features.append('component_balance_condition_code')

            
        # 5. Clear Sky Probability Features
        # Create probability-based features that indicate likelihood of clear sky
        if 'zenith' in df.columns and all(col in df.columns for col in 
                                              ['kt', 'kb', 'kd', 'direct_to_diffuse_ratio']):
            # Define zenith angle bins
            df['zenith_bin_fine'] = (df['zenith'] // 10) * 10
            self.more_categorical_features.append('zenith_bin_fine')
            # Create features that measure the likelihood of clear sky for each component
            for feature in ['kt', 'kb', 'kd', 'direct_to_diffuse_ratio']:
                # Define ideal clear sky values and allowed deviations for each feature
                if feature == 'kt':
                    ideal_value = 0.75
                    allowed_dev = 0.2
                elif feature == 'kb':
                    ideal_value = 0.85
                    allowed_dev = 0.2
                elif feature == 'kd':
                    ideal_value = 1.0
                    allowed_dev = 0.3
                elif feature == 'direct_to_diffuse_ratio':
                    ideal_value = 5.0
                    allowed_dev = 2.0
                
                # Calculate probability-like score using Gaussian function
                df[f'{feature}_clearsky_prob'] = np.exp(
                    -0.5 * ((df[feature] - ideal_value) / allowed_dev) ** 2
                )
            
            # Combine probabilities from individual features
            # Use weighted geometric mean (multiplication of probabilities with weights as exponents)
            weights = {'kt': 0.3, 'kb': 0.3, 'kd': 0.2, 'direct_to_diffuse_ratio': 0.2}
            combined_prob = np.ones(len(df))
            
            for feature, weight in weights.items():
                prob_col = f'{feature}_clearsky_prob'
                if prob_col in df.columns:
                    combined_prob *= df[prob_col] ** weight
            
            df['combined_clearsky_probability'] = combined_prob
        
        return df
    
    def _add_temporal_stability_features(self,df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal stability and relationship features"""
        
        # 1. Multi-window stability metrics
        windows = [5, 15, 30]  # minutes
        for window in windows:
            # Direct to Diffuse stability
            df[f'direct_diffuse_ratio_std_{window}'] = (
                df['direct_to_diffuse_ratio']
                .rolling(window=window, center=True)
                .std()
                .fillna(0)
            )
            
            # Component stability scores
            for comp in ['direct_normal', 'diffuse_horizontal', 'global_horizontal']:
                # Normalized standard deviation
                df[f'{comp}_norm_std_{window}'] = (
                    df[comp].rolling(window=window, center=True).std() /
                    df[comp].rolling(window=window, center=True).mean().clip(1e-6)
                ).fillna(0)
                
                # Trend consistency
                df[f'{comp}_trend_consistency_{window}'] = (
                    df[comp].rolling(window=window, center=True)
                    .apply(lambda x: np.abs(np.diff(x)).mean() / x.mean())
                    .fillna(0)
                )
    
        # 2. Component relationship features
        df['direct_diffuse_balance_score'] = (
            df['direct_to_diffuse_ratio'] / 
            df['direct_to_diffuse_ratio'].rolling(window=30, center=True).std().clip(1e-6)
        )
        
        # 3. Combined stability metrics
        for window in windows:
            # Combined component stability
            stability_features = [f'{comp}_norm_std_{window}' 
                                for comp in ['direct_normal', 'diffuse_horizontal']]
            df[f'combined_stability_{window}'] = df[stability_features].mean(axis=1)
            
            # Ratio stability scores
            ratio_features = ['direct_to_diffuse_ratio', 'diffuse_to_global_ratio']
            for ratio in ratio_features:
                df[f'{ratio}_stability_{window}'] = (
                    df[ratio] /
                    df[ratio].rolling(window=window, center=True).std().clip(1e-6)
                )
        
        # 4. Temporal pattern features
        df['direct_diffuse_trend_match'] = (
            np.sign(df['direct_normal'].diff()) == 
            np.sign(df['diffuse_horizontal'].diff())
        ).astype(int)
        
        # 5. Clear sky deviation patterns
        # for comp in ['direct_normal', 'diffuse_horizontal', 'global_horizontal']:
        #     clear_comp = f'clear_{comp.split("_")[0]}'
        #     df[f'{comp}_clear_deviation_stability'] = (
        #         (df[comp] - df[clear_comp]) /
        #         df[clear_comp].clip(1e-6)
        #     ).rolling(window=15, center=True).std().fillna(0)
        
        return df
    
    def _prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for model training or prediction, with enhanced features.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is for training (True) or prediction (False)
            
        Returns:
            Tuple of (processed DataFrame, list of feature names)
        """
        # First call the parent class implementation to get basic feature engineering
        processed_df, base_features = super()._prepare_features(df, is_training)
        
        # Apply enhanced temporal context features
        processed_df = self._add_enhanced_temporal_context(processed_df)
        
        # Apply improved direct-diffuse balance features
        processed_df = self._add_improved_direct_diffuse_balance(processed_df)
        
        processed_df = self._add_temporal_stability_features(processed_df)
        
        # Register additional categorical features
        for feature in self.more_categorical_features:
            if feature in processed_df.columns and feature not in self.categorical_features:
                self.categorical_features.append(feature)
                
        # Define enhanced feature categories
        enhanced_temporal_features = [
            # Longer rolling windows for irradiance components
            'global_horizontal_rolling_mean_15', 'global_horizontal_rolling_std_15', 'global_horizontal_coef_var_15',
            'global_horizontal_rolling_mean_20', 'global_horizontal_rolling_std_20', 'global_horizontal_coef_var_20',
            'global_horizontal_rolling_mean_30', 'global_horizontal_rolling_std_30', 'global_horizontal_coef_var_30',
            'direct_normal_rolling_mean_15', 'direct_normal_rolling_std_15', 'direct_normal_coef_var_15',
            'direct_normal_rolling_mean_20', 'direct_normal_rolling_std_20', 'direct_normal_coef_var_20',
            'direct_normal_rolling_mean_30', 'direct_normal_rolling_std_30', 'direct_normal_coef_var_30',
            'diffuse_horizontal_rolling_mean_15', 'diffuse_horizontal_rolling_std_15', 'diffuse_horizontal_coef_var_15',
            'diffuse_horizontal_rolling_mean_20', 'diffuse_horizontal_rolling_std_20', 'diffuse_horizontal_coef_var_20',
            'diffuse_horizontal_rolling_mean_30', 'diffuse_horizontal_rolling_std_30', 'diffuse_horizontal_coef_var_30',
            
            # Rate-of-change features
            'global_horizontal_max_roc_5', 'global_horizontal_stability_score_5',
            'global_horizontal_max_roc_10', 'global_horizontal_stability_score_10',
            'global_horizontal_max_roc_15', 'global_horizontal_stability_score_15',
            'direct_normal_max_roc_5', 'direct_normal_stability_score_5',
            'direct_normal_max_roc_10', 'direct_normal_stability_score_10',
            'direct_normal_max_roc_15', 'direct_normal_stability_score_15',
            'diffuse_horizontal_max_roc_5', 'diffuse_horizontal_stability_score_5',
            'diffuse_horizontal_max_roc_10', 'diffuse_horizontal_stability_score_10',
            'diffuse_horizontal_max_roc_15', 'diffuse_horizontal_stability_score_15',
            
            # Historical comparison features
            'global_horizontal_zenith_mean', 'global_horizontal_zenith_std', 'global_horizontal_zenith_zscore',
            'global_horizontal_unusual_high', 'global_horizontal_unusual_low',
            'direct_normal_zenith_mean', 'direct_normal_zenith_std', 'direct_normal_zenith_zscore',
            'direct_normal_unusual_high', 'direct_normal_unusual_low',
            'diffuse_horizontal_zenith_mean', 'diffuse_horizontal_zenith_std', 'diffuse_horizontal_zenith_zscore',
            'diffuse_horizontal_unusual_high', 'diffuse_horizontal_unusual_low',
            
            # Component correlation features
            'ghi_dni_corr_10', 'ghi_dni_corr_20', 'ghi_dni_corr_30',
            
            # Sustained clear sky features
            'sustained_clear_10', 'sustained_clear_20', 'sustained_clear_30'
        ]
        
        enhanced_direct_diffuse_features = [
            # Theoretical vs actual comparisons
            'theoretical_direct_diffuse_ratio', 'direct_diffuse_ratio_match',
            
            # Zenith-normalized diffuse analysis
            'diffuse_fraction', 'diffuse_fraction_deviation', 'zenith_adjusted_diffuse_fraction',
            
            # Consistency features
            'global_consistency_error', 'global_relative_consistency_error', 'inconsistent_measurements',
            
            # Component balance analytics
            'component_balance_condition_code', 'condition_clear_sky', 'condition_thin_clouds', 
            'condition_hazy', 'condition_cloudy',
            
            # Clear sky probability features
            'kt_clearsky_prob', 'kb_clearsky_prob', 'kd_clearsky_prob', 
            'direct_to_diffuse_ratio_clearsky_prob', 'combined_clearsky_probability'
        ]

        temporal_stability_features = [
            'direct_diffuse_ratio_std_5', 'direct_diffuse_ratio_std_15', 'direct_diffuse_ratio_std_30',
            'direct_normal_norm_std_5', 'direct_normal_norm_std_15', 'direct_normal_norm_std_30',
            'diffuse_horizontal_norm_std_5', 'diffuse_horizontal_norm_std_15', 'diffuse_horizontal_norm_std_30',
            'direct_normal_trend_consistency_5', 'direct_normal_trend_consistency_15', 'direct_normal_trend_consistency_30',
            'diffuse_horizontal_trend_consistency_5', 'diffuse_horizontal_trend_consistency_15', 'diffuse_horizontal_trend_consistency_30',
            
            'direct_diffuse_balance_score', 
            'combined_stability_5', 'combined_stability_15', 'combined_stability_30',
            'direct_to_diffuse_ratio_stability_5', 'direct_to_diffuse_ratio_stability_15', 'direct_to_diffuse_ratio_stability_30',
            'diffuse_to_global_ratio_stability_5', 'diffuse_to_global_ratio_stability_15', 'diffuse_to_global_ratio_stability_30',
            'direct_diffuse_trend_match',
            
            # 'direct_normal_clear_deviation_stability', 'diffuse_horizontal_clear_deviation_stability', 'global_horizontal_clear_deviation_stability'
        ]
        
        # Combine all potential features
        all_potential_features = base_features + enhanced_temporal_features + enhanced_direct_diffuse_features + temporal_stability_features
        
        # Filter to only include features that actually exist in the processed DataFrame
        feature_names = [f for f in all_potential_features if f in processed_df.columns]
        
        # If training, store the feature names for later use
        if is_training:
            self.feature_names = feature_names
        
        # Fix any NaN values in clear sky columns
        processed_df = fix_clear_sky_nans_with_nearby_zenith_medians(processed_df, zenith_tolerance=0.5)
        
        return processed_df, feature_names
    


    # def _add_comparative_analysis_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # """
    # Add features based on comparative analysis of true/false positive patterns.
    
    # Args:
    #     df: DataFrame with basic features
    # Returns:
    #     DataFrame with additional comparative features
    # """
    # df_copy = df.copy()
    
    # # 1. Direct-Diffuse Balance Features
    # if all(col in df_copy.columns for col in ['direct_to_diffuse_ratio', 'direct_normal', 'diffuse_horizontal']):
    #     # Ratio stability over multiple windows
    #     for window in [5, 15, 30]:
    #         # Rolling statistics of direct-diffuse ratio
    #         ratio_mean = df_copy['direct_to_diffuse_ratio'].rolling(
    #             window=window, center=True, min_periods=3
    #         ).mean()
            
    #         ratio_std = df_copy['direct_to_diffuse_ratio'].rolling(
    #             window=window, center=True, min_periods=3
    #         ).std()
            
    #         # Normalized ratio variability (targeting FP pattern)
    #         df_copy[f'ratio_stability_{window}'] = ratio_std / ratio_mean.clip(1e-6)
            
    #         # Ratio range check (FPs show wider ranges)
    #         df_copy[f'ratio_range_{window}'] = (
    #             df_copy['direct_to_diffuse_ratio'].rolling(window=window, center=True)
    #             .apply(lambda x: x.max() - x.min())
    #         )
            
    #         # Detect if ratio is in the "suspicious" range (FP common range)
    #         df_copy[f'suspicious_ratio_{window}'] = (
    #             (df_copy['direct_to_diffuse_ratio'] > 8) & 
    #             (df_copy['direct_to_diffuse_ratio'] < 12)
    #         ).astype(int)
    
    # # 2. Clearness Index Pattern Features
    # if all(col in df_copy.columns for col in ['kt', 'kb', 'kd']):
    #     # Combined clearness stability
    #     for window in [5, 15, 30]:
    #         # Track stability of all indices
    #         for idx in ['kt', 'kb', 'kd']:
    #             df_copy[f'{idx}_stability_{window}'] = (
    #                 df_copy[idx].rolling(window=window, center=True).std() /
    #                 df_copy[idx].rolling(window=window, center=True).mean().clip(1e-6)
    #             )
            
    #         # Combined stability score
    #         stability_cols = [f'{idx}_stability_{window}' for idx in ['kt', 'kb', 'kd']]
    #         df_copy[f'combined_clearness_stability_{window}'] = df_copy[stability_cols].mean(axis=1)
        
    #     # Clearness index relationships (FP patterns)
    #     df_copy['kd_kt_ratio'] = df_copy['kd'] / df_copy['kt'].clip(1e-6)
    #     df_copy['kb_kt_ratio'] = df_copy['kb'] / df_copy['kt'].clip(1e-6)
        
    #     # Detect suspicious clearness patterns
    #     df_copy['suspicious_clearness'] = (
    #         (df_copy['kd'] > 1.2) &  # High diffuse clearness
    #         (df_copy['kt'] < 1.3) &  # Moderate global clearness
    #         (df_copy['kb'] > 1.2)    # High direct clearness
    #     ).astype(int)
    
    # # 3. Component Consistency Features
    # if all(col in df_copy.columns for col in ['global_horizontal', 'direct_normal', 'diffuse_horizontal']):
    #     for window in [15, 30]:
    #         # Component variability comparison
    #         for comp in ['global_horizontal', 'direct_normal', 'diffuse_horizontal']:
    #             df_copy[f'{comp}_rel_var_{window}'] = (
    #                 df_copy[comp].rolling(window=window, center=True).std() /
    #                 df_copy[comp].rolling(window=window, center=True).mean().clip(1e-6)
    #             )
            
    #         # Relative variability ratios
    #         df_copy[f'ghi_dni_var_ratio_{window}'] = (
    #             df_copy[f'global_horizontal_rel_var_{window}'] /
    #             df_copy[f'direct_normal_rel_var_{window}'].clip(1e-6)
    #         )
            
    #         # Detect inconsistent variability patterns
    #         df_copy[f'inconsistent_variability_{window}'] = (
    #             (df_copy[f'ghi_dni_var_ratio_{window}'] > 1.5) |  # GHI more variable than DNI
    #             (df_copy[f'diffuse_horizontal_rel_var_{window}'] > 0.5)  # High diffuse variability
    #         ).astype(int)
    
    # # 4. Clear Sky Model Deviation Features
    # if all(col in df_copy.columns for col in ['clear_ghi', 'clear_dni', 'clear_dhi',
    #                                          'global_horizontal', 'direct_normal', 'diffuse_horizontal']):
    #     # Component-wise deviation from clear sky
    #     for comp, clear in zip(
    #         ['global_horizontal', 'direct_normal', 'diffuse_horizontal'],
    #         ['clear_ghi', 'clear_dni', 'clear_dhi']
    #     ):
    #         # Relative deviation
    #         df_copy[f'{comp}_clear_deviation'] = (
    #             (df_copy[comp] - df_copy[clear]) / df_copy[clear].clip(1e-6)
    #         )
            
    #         # Rolling stability of deviation
    #         for window in [15, 30]:
    #             df_copy[f'{comp}_deviation_stability_{window}'] = (
    #                 df_copy[f'{comp}_clear_deviation']
    #                 .rolling(window=window, center=True)
    #                 .std()
    #                 .fillna(0)
    #             )
        
    #     # Combined deviation pattern
    #     df_copy['suspicious_deviation_pattern'] = (
    #         (df_copy['direct_normal_clear_deviation'] > 0.2) &   # High DNI deviation
    #         (df_copy['global_horizontal_clear_deviation'] < 0.1)  # Low GHI deviation
    #     ).astype(int)
    
    # return df_copy
    


