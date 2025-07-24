import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, auc
from typing import List, Dict, Tuple, Optional, Union
from pvlib import solarposition, clearsky, atmosphere, irradiance

def fix_clear_sky_nans_with_nearby_zenith_medians(df, zenith_tolerance=0.5):
    """
    Fix NaN values in clear sky columns using median values from nearby zenith angles.
    
    Args:
        df: DataFrame with clear sky columns and zenith angles
        zenith_tolerance: Tolerance in degrees to consider for nearby zenith angles
        
    Returns:
        DataFrame with NaN values filled
    """
    # Make a copy to avoid modifying the original
    fixed_df = df.copy()
    
    # Find clear sky columns
    clear_cols = [col for col in fixed_df.columns if col.startswith('clear_') and col != 'clear_sky_score']
    
    # Check if we have zenith angles
    if 'zenith' not in fixed_df.columns:
        print("Cannot fill NaNs with zenith medians: 'zenith' column not found")
        return fixed_df
    
    # For each clear sky column
    for col in clear_cols:
        # Check if there are NaNs to fill
        nan_mask = fixed_df[col].isna()
        nan_count = nan_mask.sum()
        
        if nan_count > 0:
            print(f"Found {nan_count} NaNs in {col}")
            
            # Get unique zenith angles with NaNs
            zenith_with_nans = fixed_df.loc[nan_mask, 'zenith'].unique()
            # print(f"NaNs occur at {len(zenith_with_nans)} unique zenith angles")
            
            # For each zenith angle with NaNs
            fill_count = 0
            for zenith in zenith_with_nans:
                # Find rows with this zenith angle that have NaNs
                zenith_nan_mask = (fixed_df['zenith'] == zenith) & nan_mask
                zenith_nan_count = zenith_nan_mask.sum()
                
                # Find rows with nearby zenith angles that have valid values
                nearby_zenith_mask = (fixed_df['zenith'] >= zenith - zenith_tolerance) & \
                                    (fixed_df['zenith'] <= zenith + zenith_tolerance) & \
                                    ~fixed_df[col].isna()
                nearby_valid_values = fixed_df.loc[nearby_zenith_mask, col]
                
                if len(nearby_valid_values) > 0:
                    # Calculate median for nearby zenith angles
                    nearby_median = nearby_valid_values.median()
                    
                    # Fill NaNs at this zenith angle with the nearby median
                    fixed_df.loc[zenith_nan_mask, col] = nearby_median
                    fill_count += zenith_nan_count
                    
                    # print(f"  Filled {zenith_nan_count} NaNs at zenith={zenith:.2f}° with median={nearby_median:.2f} from {len(nearby_valid_values)} nearby values")
                else:
                    # print(f"  No valid values found near zenith={zenith:.2f}° (±{zenith_tolerance}°), cannot fill {zenith_nan_count} NaNs")
                    pass
            # Check if any NaNs remain
            remaining_nans = fixed_df[col].isna().sum()
            if remaining_nans > 0:
                # print(f"Warning: {remaining_nans} NaNs could not be filled with nearby zenith medians")
                
                # Fall back to using the overall median for the column
                if remaining_nans < nan_count:
                    overall_median = fixed_df[col].dropna().median()
                    fixed_df.loc[fixed_df[col].isna(), col] = overall_median
                    # print(f"  Filled remaining {remaining_nans} NaNs with overall median={overall_median:.2f}")
            else:
                print(f"Successfully filled all {fill_count} NaNs in {col} with nearby zenith medians")
    
    return fixed_df

class ClearSkyClassifier:
    """
    A classifier for identifying clear sky conditions based on solar irradiance data.
    Uses CatBoost algorithm with feature engineering optimized for solar data.
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
        Initialize the classifier with model parameters.
        
        Args:
            model_params: Dictionary of CatBoost parameters
            feature_engineering: Whether to perform feature engineering
            handle_imbalance: Whether to handle class imbalance
        """
        # Default model parameters
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 123,
            'eval_metric': 'F1',
            'early_stopping_rounds': 50,
            'verbose': 100
        }
        
        # Update with user-provided parameters if any
        if model_params:
            default_params.update(model_params)
            
        # Set class weights if handling imbalance
        if handle_imbalance:
            default_params['auto_class_weights'] = 'Balanced'
            
        # Initialize the model
        self.model = CatBoostClassifier(**default_params)
        
        # Store location information for clearness index calculation
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone = timezone

        # Other attributes
        self.feature_engineering = feature_engineering
        self.feature_names = []
        self.categorical_features = []
        self.is_fitted = False
        self.threshold = 0.45


    def _add_clearness_index_features(self, df):
        """
        Add clearness index features with correct timezone handling.
        
        Args:
            df: DataFrame with measured_on column and irradiance measurements
            
        Returns:
            DataFrame with added clearness index features
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        time_index = df_copy.index
        

        # Calculate solar position
        solar_position = solarposition.get_solarposition(
            time=time_index,
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
            temperature=df_copy.get('temperature', 20)
        )
        # Add solar position and clear sky data to the dataframe
        df_copy['zenith'] = solar_position['apparent_zenith']
        df_copy['elevation'] = solar_position['apparent_elevation']
        df_copy['azimuth'] = solar_position['azimuth']

        # Get airmass
        airmass = atmosphere.get_relative_airmass(solar_position['apparent_zenith'])
        
        # Get Linke turbidity
        linke_turbidity = clearsky.lookup_linke_turbidity(
            time=time_index,
            latitude=self.latitude,
            longitude=self.longitude
        )
        
        # Ensure linke_turbidity has the same timezone as time_index
        # if hasattr(linke_turbidity, 'index') and linke_turbidity.index.tz != time_index.tz:
        #     if linke_turbidity.index.tz is None:
        #         # If linke_turbidity index is naive, localize it
        #         linke_turbidity.index = linke_turbidity.index.tz_localize(time_index.tz)
        #     else:
        #         # If it has a different timezone, convert it
        #         linke_turbidity.index = linke_turbidity.index.tz_convert(time_index.tz)

        # Calculate clear sky irradiance
        clear_sky = clearsky.ineichen(
            solar_position['apparent_zenith'],
            airmass,
            linke_turbidity,
            altitude=self.altitude,
            dni_extra=irradiance.get_extra_radiation(time_index)
        )
        
        # Calculate clearness indices
        df_copy['kt'] = df_copy['global_horizontal'].values / clear_sky['ghi'].values
            
        df_copy['kb'] = df_copy['direct_normal'].values / clear_sky['dni'].values
            
        df_copy['kd'] = df_copy['diffuse_horizontal'].values / clear_sky['dhi'].values
        
        # Add the clear sky model values
        # Shift clear sky values to align with clock time measurements
        # This accounts for the difference between solar time and clock time
        # (Solar noon occurs at 13:01 local time during DST)
        shift_minutes = -60  # Approximate shift needed
        df_copy['clear_ghi'] = clear_sky['ghi'].shift(shift_minutes)
        df_copy['clear_dni'] = clear_sky['dni'].shift(shift_minutes)
        df_copy['clear_dhi'] = clear_sky['dhi'].shift(shift_minutes)
    

        # Add solar position features
        df_copy['zenith'] = solar_position['apparent_zenith'].values
        df_copy['azimuth'] = solar_position['azimuth'].values
        
        # Add diffuse enhancement ratio to captures the "haziness" factor
        if 'diffuse_horizontal' in df_copy.columns:
            df_copy['diffuse_enhancement'] = np.divide(
                df_copy['diffuse_horizontal'].values,
                clear_sky['dhi'].values,
                out=np.zeros_like(clear_sky['dhi'].values),
                where=clear_sky['dhi'].values > 0
            )
            
            # Clip to reasonable values
            df_copy['diffuse_enhancement'] = df_copy['diffuse_enhancement'].clip(0, 5.0)
            
            # Create a categorical feature for diffuse enhancement
            conditions = [
                (df_copy['diffuse_enhancement'] <= 1.1),  # Normal clear sky
                (df_copy['diffuse_enhancement'] <= 1.5),  # Hazy clear sky
                (df_copy['diffuse_enhancement'] > 1.5)    # Cloudy or very hazy
            ]
            choices = [0, 1, 2]
            df_copy['diffuse_category'] = np.select(conditions, choices, default=2)
        
        
        # Handle potential infinity and NaN values
        for col in ['kt', 'kb', 'kd']:
            if col in df_copy.columns:
                # Cap extremely high values (can happen at very low irradiance)
                df_copy[col] = df_copy[col].clip(0, 2.0)
                # Fill NaN values (can happen at night)
                df_copy[col] = df_copy[col].fillna(0)
        
    
        return df_copy


    def _add_advanced_clearness_features(self, df):
        """
        Add advanced features derived from clearness indices.
        
        Args:
            df: DataFrame with basic clearness indices already calculated
            
        Returns:
            DataFrame with additional advanced clearness features
        """
        
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Ensure the basic clearness indices exist
        required_cols = ['kt', 'kb', 'kd']
        missing = [col for col in required_cols if col not in df_copy.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Run add_clearness_index_features first.")
        
        # Clearness stability (rolling standard deviation)
        # Lower values indicate more stable clear conditions
        window_sizes = [3, 5, 9]  # Different window sizes for different time scales
        
        for window in window_sizes:
            # Global clearness stability
            df_copy[f'kt_stability_{window}'] = df_copy['kt'].rolling(window, center=True).std().fillna(0)
            
            # Direct clearness stability
            df_copy[f'kb_stability_{window}'] = df_copy['kb'].rolling(window, center=True).std().fillna(0)
            
            # Diffuse clearness stability
            df_copy[f'kd_stability_{window}'] = df_copy['kd'].rolling(window, center=True).std().fillna(0)
        
        # Clearness ratios
        # Ratio of direct to diffuse clearness - higher in clear conditions
        df_copy['kb_to_kd_ratio'] = (df_copy['kb'] / df_copy['kd']).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Normalized clearness
        # Normalize by the maximum clearness observed at similar solar zenith angles
        if 'zenith' in df_copy.columns:
            # Bin zenith angles by 5 degrees
            df_copy['zenith_bin'] = (df_copy['zenith'] // 5) * 5
            
            # Calculate maximum kt for each zenith bin
            zenith_max_kt = df_copy.groupby('zenith_bin')['kt'].transform('max')
            df_copy['kt_normalized'] = df_copy['kt'] / zenith_max_kt.replace(0, 1)
            
            # Same for direct clearness
            zenith_max_kb = df_copy.groupby('zenith_bin')['kb'].transform('max')
            df_copy['kb_normalized'] = df_copy['kb'] / zenith_max_kb.replace(0, 1)
        
        # Clearness deviation from expected clear sky pattern
        # Calculate how much the clearness deviates from a smooth curve
        for col in ['kt', 'kb']:
            # Use a larger window to get the expected smooth pattern
            df_copy[f'{col}_smooth'] = df_copy[col].rolling(15, center=True).mean().fillna(df_copy[col])
            df_copy[f'{col}_deviation'] = np.abs(df_copy[col] - df_copy[f'{col}_smooth'])
        
        # Combined clearness score
        # A single feature that combines multiple clearness indicators
        # Higher values indicate clearer conditions
        df_copy['clear_sky_score'] = (
            df_copy['kt'] * 0.4 +                  # Global clearness
            df_copy['kb'] * 0.4 +                  # Direct clearness
            (1 - df_copy['kt_stability_5']) * 0.1 + # Stability of global clearness
            (1 - df_copy['kb_stability_5']) * 0.1   # Stability of direct clearness
        )
        # clear but hazy
        df_copy['clear_but_hazy'] = ((df_copy['kt'] > 0.8) & 
                                  (df_copy['kb'] > 0.8) & 
                                  (df_copy['kd'] > 1.2)).astype(int)
        # cloud 
        df_copy['cloud_enhancement'] = (df_copy['kt'] > 1.0).astype(int)
        
        # Severity of cloud enhancement
        df_copy['cloud_enhancement_severity'] = np.clip((df_copy['kt'] - 1.0) * 5, 0, 1)
        
        # Pattern from false positives: high kb with moderate kt
        df_copy['partial_cloud_pattern'] = ((df_copy['kb'] > 0.8) & 
                                            (df_copy['kt'] < 0.5)).astype(int)
    
        # Low sun angle and dawn/dusk features
        df_copy['low_sun_angle'] = (df_copy['zenith'] > 75).astype(int)
        df_copy['cos_zenith'] = np.cos(np.radians(df_copy['zenith']))

        df_copy['dawn_dusk'] = ((df_copy['hour'] <= 8) | 
                                (df_copy['hour'] >= 17)).astype(int)
                                
        return df_copy
        
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from the raw data.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with engineered features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure datetime column is properly formatted
        if 'measured_on' in result.columns and not pd.api.types.is_datetime64_any_dtype(result['measured_on']):
            result['measured_on'] = pd.to_datetime(result['measured_on'])
        
        # Extract temporal features
        if 'measured_on' in result.columns:
            # Hour of day (cyclical encoding)
            result['hour'] = result['measured_on'].dt.hour
            result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
            result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
            
            # Day of year (cyclical encoding)
            result['day_of_year'] = result['measured_on'].dt.dayofyear
            result['day_sin'] = np.sin(2 * np.pi * result['day_of_year'] / 365)
            result['day_cos'] = np.cos(2 * np.pi * result['day_of_year'] / 365)
            
            # Month as categorical
            result['month'] = result['measured_on'].dt.month
            self.categorical_features.append('month')
            
            # Is weekend
            result['is_weekend'] = result['measured_on'].dt.dayofweek >= 5
            self.categorical_features.append('is_weekend')
        
        # Create ratio features (handling division by zero)
        if all(col in result.columns for col in ['direct_normal', 'global_horizontal']):
            result['direct_to_global_ratio'] = result['direct_normal'] / result['global_horizontal'].replace(0, np.nan)
            result['direct_to_global_ratio'].fillna(0, inplace=True)
        
        if all(col in result.columns for col in ['diffuse_horizontal', 'global_horizontal']):
            result['diffuse_to_global_ratio'] = result['diffuse_horizontal'] / result['global_horizontal'].replace(0, np.nan)
            result['diffuse_to_global_ratio'].fillna(0, inplace=True)
        
        if all(col in result.columns for col in ['direct_normal', 'diffuse_horizontal']):
            result['direct_to_diffuse_ratio'] = result['direct_normal'] / result['diffuse_horizontal'].replace(0, np.nan)
            result['direct_to_diffuse_ratio'].fillna(0, inplace=True)
        
        # Create binary feature for daytime
        if 'global_horizontal' in result.columns:
            result['is_daytime'] = result['global_horizontal'] > 5  # Threshold can be adjusted
        
        # Statistical features (if time series data is available)
        if 'measured_on' in result.columns and result['measured_on'].is_monotonic_increasing:
            for col in ['direct_normal', 'global_horizontal', 'diffuse_horizontal']:
                if col in result.columns:
                    # Rolling mean (10-minute window = 10 samples at 1-min intervals)
                    result[f'{col}_rolling_mean'] = result[col].rolling(window=10, min_periods=1).mean()
                    
                    # Rolling standard deviation (measure of stability)
                    result[f'{col}_rolling_std'] = result[col].rolling(window=10, min_periods=1).std().fillna(0)
                    
                    # Rate of change
                    result[f'{col}_rate_of_change'] = result[col].diff().fillna(0)
        
        return result
    
    def _prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for model training or prediction.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is for training (True) or prediction (False)
            
        Returns:
            Tuple of (processed DataFrame, list of feature names)
        """
        # Apply feature engineering if enabled
        if self.feature_engineering:
            processed_df = self._engineer_features(df)
            if self.latitude is not None and self.longitude is not None:
                processed_df = self._add_clearness_index_features(processed_df)
                processed_df = self._add_advanced_clearness_features(processed_df)     
        else:
            processed_df = df.copy()
        
       
        base_features = ['direct_normal', 'global_horizontal', 'diffuse_horizontal']
        
        # Add engineered features 
        engineered_features = [
            'hour', 'hour_sin', 'hour_cos', 
            # 'day_sin', 'day_cos', 
            'month', 'is_weekend',
            'direct_to_global_ratio', 'diffuse_to_global_ratio', 'direct_to_diffuse_ratio',
            'is_daytime'
        ]
        
        # Add statistical features 
        stat_features = [
            'direct_normal_rolling_mean', 'direct_normal_rolling_std', 'direct_normal_rate_of_change',
            'global_horizontal_rolling_mean', 'global_horizontal_rolling_std', 'global_horizontal_rate_of_change',
            'diffuse_horizontal_rolling_mean', 'diffuse_horizontal_rolling_std', 'diffuse_horizontal_rate_of_change'
        ]
        
        # Add clearness features
        clearness_features = [
            'kt', 'kb', 'kd',
            'clear_ghi', 'clear_dni', 'clear_dhi',
            'zenith', 
            'azimuth', 
            'elevation',
            'kt_stability_3', 'kt_stability_5', 'kt_stability_9',
            'kb_stability_3', 'kb_stability_5', 'kb_stability_9',
            'kd_stability_3', 'kd_stability_5', 'kd_stability_9',
            'kb_to_kd_ratio',
            'kt_normalized', 'kb_normalized',
            'kt_deviation', 'kb_deviation',
            'clear_sky_score', 'diffuse_enhancement', 'diffuse_category',
            'clear_but_hazy', 'cloud_enhancement', 'cloud_enhancement_severity',
            'partial_cloud_pattern', 'low_sun_angle', 'dawn_dusk'
        ]
        # Combine all potential features
        all_potential_features = base_features + engineered_features + stat_features + clearness_features
        
        feature_names = [f for f in all_potential_features if f in processed_df.columns]
        
        # If training, store the feature names for later use
        if is_training:
            self.feature_names = feature_names
        
        processed_df = fix_clear_sky_nans_with_nearby_zenith_medians(processed_df, zenith_tolerance=0.5)
        return processed_df, feature_names
    
    def fit(self, 
            X: pd.DataFrame, 
            y: Union[pd.Series, np.ndarray],
            eval_set: Optional[Tuple[pd.DataFrame, Union[pd.Series, np.ndarray]]] = None,
            **kwargs) -> 'ClearSkyClassifier':
        """
        Fit the model to the training data.
        
        Args:
            X: Training features
            y: Target variable (clearsky_label)
            eval_set: Optional evaluation set for early stopping
            **kwargs: Additional arguments to pass to CatBoost's fit method
            
        Returns:
            Self for method chaining
        """
        # Prepare features
        X_processed, features = self._prepare_features(X, is_training=True)
        
        # Prepare evaluation set if provided
        if eval_set is not None:
            eval_X, eval_y = eval_set
            eval_X_processed, _ = self._prepare_features(eval_X, is_training=False)
            eval_set = [(eval_X_processed[features], eval_y)]
        
        # Create CatBoost pool with categorical features
        train_pool = Pool(
            data=X_processed[features],
            label=y,
            cat_features=[i for i, f in enumerate(features) if f in self.categorical_features]
        )
        
        # Fit the model
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            **kwargs
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Prepare features
        X_processed, _ = self._prepare_features(X, is_training=False)
        
        # Make predictions
        predictions = self.model.predict(X_processed[self.feature_names])

        # # Get class probabilities
        # probas = self.predict_proba(X)
        
        # # Apply custom threshold to the positive class probability
        # predictions = (probas[:, 1] >= self.threshold).astype(int)
        
        return predictions

    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Prepare features
        X_processed, _ = self._prepare_features(X, is_training=False)
        
        # Make probability predictions
        return self.model.predict_proba(X_processed[self.feature_names])
    
    def evaluate(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Make predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        report = classification_report(y, y_pred, output_dict=True)
        
        # Calculate precision-recall AUC
        precision, recall, _ = precision_recall_curve(y, y_proba)
        pr_auc = auc(recall, precision)
        
        # Combine metrics
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
            'pr_auc': pr_auc
        }
        
        return metrics
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to show
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Get feature importance
        feature_importance = self.model.get_feature_importance()
        feature_names = self.feature_names
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:
        """
        Plot confusion matrix.
        
        Args:
            X: Test features
            y: True labels
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Clear Sky', 'Clear Sky'],
                    yticklabels=['Not Clear Sky', 'Clear Sky'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:
        """
        Plot precision-recall curve.
        
        Args:
            X: Test features
            y: True labels
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Make probability predictions
        y_proba = self.predict_proba(X)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y, y_proba)
        pr_auc = auc(recall, precision)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str) -> 'ClearSkyClassifier':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        self.model.load_model(filepath)
        self.is_fitted = True
        return self

