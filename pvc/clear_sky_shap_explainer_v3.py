import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
import seaborn as sns
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.gridspec as gridspec
import io
import tempfile
from matplotlib.image import imread
import os
from datetime import datetime, timedelta

class ClearSkyShapExplainer:
    """
    A class for explaining ClearSkyClassifier predictions using SHAP values.
    Provides various visualization and analysis methods to understand model behavior.
    """
    
    def __init__(self, model: 'ClearSkyClassifier'):
        """
        Initialize the explainer with a trained ClearSkyClassifier model.
        
        Args:
            model: Trained ClearSkyClassifier model
        """
        if not model.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        self.model = model
        self.explainer = None
        self.feature_names = model.feature_names
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the SHAP TreeExplainer for the model."""
        self.explainer = shap.TreeExplainer(self.model.model)
        
    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for SHAP analysis using the model's feature preparation.
        
        Args:
            X: Input features
            
        Returns:
            Processed DataFrame with the correct features
        """
        # print("\n=== PREPARING DATA FOR SHAP ANALYSIS ===")
        # print(f"Input data shape: {X.shape}")
        
        # Sort data by time if needed
        has_measured_on = 'measured_on' in X.columns
        is_monotonic = has_measured_on and X['measured_on'].is_monotonic_increasing
        
        if has_measured_on and not is_monotonic:
            print("Sorting data by 'measured_on' to ensure rolling features are generated")
            X_sorted = X.sort_values('measured_on').reset_index(drop=True)
        else:
            X_sorted = X.copy()
        
        # Use the model's feature preparation with sorted data
        X_processed, _ = self.model._prepare_features(X_sorted, is_training=False)
        
        # Check for NaN values in the processed data
        nan_counts = X_processed.isna().sum()
        # print("\nNaN counts in processed data:")
        # print(nan_counts[nan_counts > 0])  # Only show columns with NaNs
        
        # Get the model's internal feature names
        model_feature_names = self.model.model.feature_names_
        
        # Create DataFrame with features in the correct order
        X_for_shap = pd.DataFrame()
        for feature in model_feature_names:
            if feature in X_processed.columns:
                X_for_shap[feature] = X_processed[feature]
            else:
                # print(f"WARNING: Feature {feature} is missing and will be filled with zeros")
                X_for_shap[feature] = 0
        
        # Handle missing values appropriately for each feature type
        # print("\nHandling missing values:")
        
        # 1. For categorical features: fill with a special category
        categorical_features = ['month', 'is_weekend', 'diffuse_category']
        for feature in categorical_features:
            if feature in X_for_shap.columns:
                nan_count = X_for_shap[feature].isna().sum()
                if nan_count > 0:
                    # print(f"  - Filling {nan_count} NaNs in {feature} with 'missing'")
                    X_for_shap[feature] = X_for_shap[feature].fillna('missing')
                # Ensure categorical features are strings
                X_for_shap[feature] = X_for_shap[feature].astype(str)
        
        # 2. For ratio features: fill with 0 (assuming 0/x = 0)
        ratio_features = [f for f in X_for_shap.columns if 'ratio' in f.lower()]
        for feature in ratio_features:
            nan_count = X_for_shap[feature].isna().sum()
            if nan_count > 0:
                # print(f"  - Filling {nan_count} NaNs in {feature} with 0")
                X_for_shap[feature] = X_for_shap[feature].fillna(0)
        
        # 3. For rolling features: forward fill then backfill (maintain time patterns)
        rolling_features = [f for f in X_for_shap.columns if 'rolling' in f.lower()]
        for feature in rolling_features:
            nan_count = X_for_shap[feature].isna().sum()
            if nan_count > 0:
                # print(f"  - Filling {nan_count} NaNs in {feature} with ffill/bfill")
                X_for_shap[feature] = X_for_shap[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 4. For remaining features: fill with 0
        remaining_features = [f for f in X_for_shap.columns 
                            if f not in categorical_features + ratio_features + rolling_features]
        for feature in remaining_features:
            nan_count = X_for_shap[feature].isna().sum()
            if nan_count > 0:
                # print(f"  - Filling {nan_count} NaNs in {feature} with 0")
                X_for_shap[feature] = X_for_shap[feature].fillna(0)
        
        # Final check for any remaining NaNs
        remaining_nans = X_for_shap.isna().sum().sum()
        if remaining_nans > 0:
            # print(f"WARNING: {remaining_nans} NaN values still remain. Filling all with 0.")
            X_for_shap = X_for_shap.fillna(0)
        else:
            # print("All NaN values have been handled.")
            pass
        
        # print(f"Final data shape for SHAP: {X_for_shap.shape}")
        
        return X_for_shap
    

    def compute_shap_values(self, X: pd.DataFrame, sample_size: Optional[int] = None) -> Any:
        """
        Compute SHAP values for the given data.
        
        Args:
            X: Input features
            sample_size: Optional sample size to limit computation for large datasets
            
        Returns:
            SHAP values object
        """
        print("Computing SHAP values for", len(X) if sample_size is None else sample_size, "samples")
        
        # Time-preserving sampling
        if sample_size is not None and len(X) > sample_size:
            # Sort by time first
            X_sorted = X.sort_values('measured_on')
            # Take evenly spaced samples to preserve time patterns
            step = len(X_sorted) // sample_size
            X_sample = X_sorted.iloc[::step][:sample_size]
        else:
            X_sample = X
        
        # Debug: Print input data information
        print(f"Input data shape: {X_sample.shape}")
        print(f"Input data columns: {X_sample.columns.tolist()}")
        
        # Prepare data
        X_processed = self._prepare_data(X_sample)
        
        # Debug: Check the model's internal feature names
        print("\n=== MODEL'S INTERNAL FEATURE INFORMATION ===")
        
        # For CatBoost models, try to get feature names directly
        print("CatBoost model feature names:")
        try:
            # This gets the feature names as used by the model internally
            feature_names = self.model.model.feature_names_
            print(f"Model's feature_names_: {feature_names}")
        except:
            print("Could not access model.feature_names_")
        
        # Debug: Print processed data information
        print(f"\nX_processed shape before SHAP: {X_processed.shape}")
        print(f"First few columns: {X_processed.columns[:5].tolist()}")
        
        # Handle categorical features
        # categorical_features = ['month', 'is_weekend', 'diffuse_category']
        # boolean_features = ['is_daytime', 'clear_but_hazy', 'cloud_enhancement', 
        #                 'partial_cloud_pattern', 'low_sun_angle', 'dawn_dusk']
        
        # for feature in categorical_features:
        #     if feature in X_processed.columns:
        #         print(f"Converting {feature} to string to ensure it's treated as categorical")
        #         X_processed[feature] = X_processed[feature].astype(str)
        
        # for feature in boolean_features:
        #     if feature in X_processed.columns:
        #         print(f"Converting {feature} to int to ensure it's treated as numeric")
        #         X_processed[feature] = X_processed[feature].astype(int)
        
        # print(f"X_processed shape after processing: {X_processed.isnull().sum()}")
        # print(f"X_processed columns: {X_processed.columns.tolist()}")
        # Return SHAP values
        return self.explainer(X_processed)

    def plot_summary(self, X: pd.DataFrame, sample_size: int = 100, max_display: int = 20):
        """
        Plot a summary of SHAP values for the model.
        
        Args:
            X: Features
            sample_size: Number of samples to use for SHAP values
            max_display: Maximum number of features to display
        """
        # Compute SHAP values
        shap_values = self.compute_shap_values(X, sample_size)
        
        # Get the data used for the plot
        # Since we're sampling from X in compute_shap_values, we need to use the same subset
        if len(X) > sample_size:
            # If we sampled, use the first sample_size rows
            X_subset = X.iloc[:sample_size]
        else:
            # If we didn't sample, use all of X
            X_subset = X
        
        # Prepare the data for plotting
        data_for_plot = self._prepare_data(X_subset)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values.values, 
                        data_for_plot, 
                        max_display=max_display,
                        show=False)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance(self, X: pd.DataFrame, max_display: int = 20, sample_size: Optional[int] = 100):
        """
        Plot feature importance based on SHAP values.
        
        Args:
            X: Input features
            max_display: Maximum number of features to display
            sample_size: Optional sample size to limit computation
        """
        shap_values = self.compute_shap_values(X, sample_size)
        
        plt.figure(figsize=(10, 8))
        shap.plots.bar(shap_values, max_display=max_display, show=False)

        # Y-axis labels
        plt.yticks(fontsize=10)

        # X-axis labels and title
        plt.xticks(fontsize=10)
        plt.xlabel("mean(|SHAP value|)", fontsize=10)
        plt.title("SHAP Feature Importance", fontsize=14)

        # Adjust tick parameters for better visibility
        plt.tick_params(axis='y', which='major', labelsize=10, length=6, width=1)

        # Add grid for better readability
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
    
    def plot_dependence(self, X: pd.DataFrame, feature: Union[str, int], 
                    interaction_feature: Optional[Union[str, int]] = "auto",
                    highlight_fp=True, highlight_fn=False,
                    sample_size: Optional[int] = 100):
        """
        Create a dependence plot for a specific feature.
        
        Args:
            X: Input features
            feature: Feature to analyze (name or index)
            interaction_feature: Feature to use for coloring (name, index, or "auto")
            sample_size: Optional sample size to limit computation
        """
        # Compute SHAP values
        shap_values = self.compute_shap_values(X, sample_size)
        
        # Get the data used for the plot
        # Since we're sampling from X in compute_shap_values, we need to use the same subset
        if len(X) > sample_size:
            # If we sampled, use the first sample_size rows
            X_subset = X.iloc[:sample_size]
        else:
            # If we didn't sample, use all of X
            X_subset = X
        
        # Prepare the data for plotting
        X_processed = self._prepare_data(X_subset)
        
        # Convert feature name to index if needed
        if isinstance(feature, str):
            if feature in self.feature_names:
                feature_idx = self.feature_names.index(feature)
            else:
                raise ValueError(f"Feature '{feature}' not found in model features")
        else:
            feature_idx = feature
        
        # Convert interaction feature name to index if needed
        if isinstance(interaction_feature, str) and interaction_feature != "auto":
            if interaction_feature in self.feature_names:
                interaction_idx = self.feature_names.index(interaction_feature)
            else:
                raise ValueError(f"Interaction feature '{interaction_feature}' not found in model features")
        else:
            interaction_idx = interaction_feature
        
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(feature_idx, 
                             shap_values.values, 
                             X_processed, 
                             interaction_index=interaction_idx,
                             feature_names=self.feature_names,
                             show=False)
        plt.tight_layout()
        plt.show()
        
    def plot_force(self, X: pd.DataFrame, sample_idx: int = 0, y: pd.Series = None):
        """
        Create a force plot for a specific sample using the SHAP library.
        
        Args:
            X: Input features
            sample_idx: Index of the sample to explain
        """
        # Get a single sample
        X_single = X.iloc[[sample_idx]]
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X_single)
        
        # Prepare the data for plotting
        # X_processed = self._prepare_data(X_single)
        
        # Get the feature values and names
        # feature_values = X_processed.iloc[0].values
        
        # plt.figure(figsize=(20, 7))
        # shap.plots.force(
        #     base_value=shap_values.base_values[0],
        #     shap_values=shap_values.values[0],
        #     features=feature_values,
        #     feature_names=self.feature_names,
        #     matplotlib=True,
        #     show=False,
        # )
        # plt.title(f"SHAP Force Plot for Sample {sample_idx}", fontsize=16, pad=20, y=2.0)
        # plt.tight_layout()
        # plt.show()
        title = f"SHAP Force Plot for Sample {sample_idx}"
        self._plot_force_with_rounded_values(shap_values, sample_idx, title, y_true=y[sample_idx])

    def plot_multiple_forces_v1(self, X: pd.DataFrame, sample_indices: List[int], y: pd.Series,
                        n_cols: int = 1, figsize: Optional[Tuple[int, int]] = None):
        """
        Create force plots for multiple samples using the SHAP library.
        Uses an image-based approach to organize force plots in a clean grid layout.
        
        Args:
            X: Input features DataFrame
            sample_indices: List of indices of samples to explain
            y: True labels
            n_cols: Number of columns in the grid layout
            figsize: Figure size (width, height) in inches, default is auto-calculated
        
        Returns:
            Matplotlib figure with force plots organized in a grid
        """
        
        # Close any existing plots to avoid duplicates
        plt.close('all')
        
        n_samples = len(sample_indices)
        if n_samples == 0:
            print("No samples provided to plot.")
            return None
        
        # Calculate grid dimensions
        n_cols = min(n_cols, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols  # Ceiling division
        
        # Create temporary directory for individual plots
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {temp_dir}")
        
        # Store information about each plot
        plot_info = {}
        
        # Compute SHAP values once for all samples
        shap_values = self.compute_shap_values(X)
        X_processed = self._prepare_data(X, is_training=False)
        all_probs = self.model.predict_proba(X_processed)[:, 1]
        
        # Create individual force plots
        for i, sample_idx in enumerate(sample_indices):
            plot_key = f"sample_{i}"
            
            # Skip if index is out of bounds
            if sample_idx >= len(X):
                print(f"Sample index {sample_idx} is out of bounds. Skipping.")
                plot_info[plot_key] = {
                    'status': 'error',
                    'message': f"Index {sample_idx} out of bounds"
                }
                continue
            
            print(f"\nProcessing sample {sample_idx}:")
            print(f"Original probability: {all_probs[sample_idx]:.3f}")
            
            # Get sample information
            sample_timestamp = X.index[sample_idx]
            title = f"Sample {sample_idx}: {sample_timestamp}"
            
            # Create a new figure for this force plot
            # plt.figure(figsize=(8, 2))  # Force plots are wider than tall
            # figsize = (20, 3)
            self._plot_force_with_rounded_values(shap_values, 
                                                 sample_idx, 
                                                 title, 
                                                 y_true=y[sample_idx])
            
            # Save to file
            plot_file = os.path.join(temp_dir, f"{plot_key}.png")
            plt.savefig(plot_file, dpi=200, bbox_inches='tight')
            plt.close()
            
            # Store plot info
            plot_info[plot_key] = {
                'status': 'created',
                'file': plot_file,
                'sample_idx': sample_idx,
                'timestamp': sample_timestamp
            }
        
        # Create figure for the grid of plots
        if not figsize:
            # Auto-calculate figure size based on grid dimensions
            figsize = (n_cols * 20, n_rows * 3)
            
        print(f"Creating plot grid with figsize: {figsize}")
        # Create the main figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle case of single row or column
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])
        
        # Fill the grid with plots
        for i in range(n_samples):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            plot_key = f"sample_{i}"
            
            if plot_key in plot_info and plot_info[plot_key]['status'] == 'created':
                # Load and display the saved image
                img = imread(plot_info[plot_key]['file'])
                ax.imshow(img)
                ax.axis('off')
            elif plot_key in plot_info:
                # Display error message
                message = plot_info[plot_key].get('message', 'Unknown issue')
                ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=10,
                    transform=ax.transAxes)
                ax.set_facecolor('#f8f8f8')
                ax.axis('off')
            else:
                # Empty subplot
                ax.set_facecolor('#f8f8f8')
                ax.axis('off')
        
        # Hide any unused subplots
        for i in range(n_samples, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        # Add overall title
        plt.suptitle(f"SHAP Force Plots for {n_samples} Samples", fontsize=14, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
        
        # Clean up temporary files
        for info in plot_info.values():
            if info['status'] == 'created' and os.path.exists(info['file']):
                os.remove(info['file'])
        
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        # Display the figure
        plt.show()
        return fig

    def plot_multiple_forces(self, X: pd.DataFrame, sample_indices: List[int], y: pd.Series):
        """
        Create force plots for multiple samples using the SHAP library.
        
        Args:
            X: Input features DataFrame
            sample_indices: List of indices of samples to explain
            n_cols: Number of columns in the grid layout
            figsize: Figure size (width, height) in inches, default is auto-calculated
        """
        n_samples = len(sample_indices)
        if n_samples == 0:
            print("No samples provided to plot.")
            return
        
        shap_values = self.compute_shap_values(X)
        X_processed = self._prepare_data(X)
        all_probs = self.model.predict_proba(X_processed)[:, 1]
    
        # Process each sample
        for i, sample_idx in enumerate(sample_indices):
            # Skip if index is out of bounds
            if sample_idx >= len(X):
                print(f"Sample index {sample_idx} is out of bounds. Skipping.")
                continue
            
            print(f"\nProcessing sample {sample_idx}:")
            print(f"Original probability: {all_probs[sample_idx]:.3f}")
            
            
            # Add sample information as title
            sample_timestamp = X.index[sample_idx]
            title = f"SHAP Force Plot for Sample {sample_idx}: {sample_timestamp}"
            self._plot_force_with_rounded_values(shap_values, sample_idx, title, y_true=y[sample_idx])

        
        plt.tight_layout()
        plt.show()
        
        return plt

    def plot_multiple_forces_from_cluster(self, X: pd.DataFrame, y: pd.Series, cluster_results: dict, 
                            cluster_id: int, n_samples: int = 6, seed: int = 41):
        """
        Plot force plots for random samples from a specific cluster.
        
        Args:
            X: Input features DataFrame
            y: True labels
            cluster_results: Dictionary containing clustering results including:
                            - cluster_labels
                            - original_indices (or other indices)
            cluster_id: ID of the cluster to sample from
            n_samples: Number of samples to plot
            seed: Random seed for reproducibility
        """
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Extract cluster data
        cluster_labels = cluster_results['cluster_labels']
        cluster_data = cluster_results['data']
        
        # Get datetime indices if available
        datetime_indices = cluster_results.get('datetime_indices', [])
        
        # Find indices of samples in the specified cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_positions = np.where(cluster_mask)[0]
        
        print(f"\nCluster {cluster_id} Analysis:")
        print(f"Total samples in cluster: {len(cluster_positions)}")
        
        # Sample from the cluster positions
        if len(cluster_positions) <= n_samples:
            selected_positions = cluster_positions
            print(f"Using all {len(selected_positions)} samples from cluster {cluster_id}")
        else:
            selected_positions = np.random.choice(cluster_positions, size=n_samples, replace=False)
            print(f"Selected {n_samples} random samples from cluster {cluster_id} (total: {len(cluster_positions)} samples)")
        
        # Map positions to datetime indices if available
        if len(datetime_indices) > 0 and all(pos < len(datetime_indices) for pos in selected_positions):
            cluster_indices = [datetime_indices[pos] for pos in selected_positions]
        else:
            # Fallback to using cluster_data index
            cluster_samples = cluster_data[cluster_data['cluster'] == cluster_id]
            cluster_indices = cluster_samples.index.tolist()
            
            # If more samples needed than available in DataFrame, take a random subset
            if len(selected_positions) > len(cluster_indices):
                if len(cluster_indices) == 0:
                    print("No samples available in cluster data")
                    return None
                # Use available indices with replacement if needed
                cluster_indices = np.random.choice(cluster_indices, size=min(n_samples, len(cluster_indices)), replace=len(cluster_indices) < n_samples)
        
        # Find valid indices that exist in X
        valid_indices = [idx for idx in cluster_indices if idx in X.index]
        
        if len(valid_indices) == 0:
            print("No valid indices found in dataset X. Cannot create plots.")
            return None
        
        print(f"Found {len(valid_indices)} valid indices in the dataset")
        
        # Print sample information
        print("\nSelected samples:")
        for i, idx in enumerate(valid_indices):
            print(f"\nSample {i+1}:")
            print(f"Index: {idx}")
            
            # Print key features if available
            key_features = ['direct_normal', 'global_horizontal', 'diffuse_horizontal', 
                        'kt', 'kb', 'kd', 'clear_sky_score']
            available_features = [f for f in key_features if f in X.columns]
            for feature in available_features:
                print(f"{feature}: {X.loc[idx][feature]:.4f}")
        
        # Get predictions for verification
        X_processed = self._prepare_data(X)
        y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
        
        print("\nPrediction verification for selected samples:")
        for i, idx in enumerate(valid_indices):
            # Find position in X
            position = X.index.get_loc(idx)
            
            print(f"\nSample {i+1} (Index: {idx}):")
            print(f"Prediction probability: {y_pred_proba[position]:.3f}")
            y_value = y.loc[idx] if hasattr(y, 'loc') else y[position]
            print(f"True label: {y_value}")
            misclass_type = "False Positive" if y_value == 0 and y_pred_proba[position] >= 0.5 else "False Negative"
            print(f"Classification: {misclass_type}")
        
        # Plot the selected samples using force plots
        return self.plot_multiple_forces(X, valid_indices, y)

    def plot_decision(self, X: pd.DataFrame, sample_indices: List[int] = None, feature_names: List[str] = None, 
                    feature_order: str = "importance", plot_height: float = 0.4):
        """
        Create a SHAP decision plot for multiple samples.
        
        Args:
            X: Input features DataFrame
            sample_indices: Indices of samples to include (defaults to all samples if None)
            feature_names: List of specific features to include (defaults to all if None)
            feature_order: How to order features ('importance', 'hclust', or a list of feature names)
            plot_height: Height of the plot in inches per feature
        """
        # Use all samples if none specified
        if sample_indices is None:
            sample_indices = list(range(min(10, len(X))))  # Default to first 10 samples
        
        # Select the samples
        X_subset = X.iloc[sample_indices]
        
        # Prepare the data
        X_processed = self._prepare_data(X_subset)
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X_subset)
        
        # Use specified feature names or all feature names
        if feature_names is not None:
            feature_mask = [i for i, name in enumerate(self.feature_names) if name in feature_names]
            plot_features = feature_names
        else:
            feature_mask = None
            plot_features = self.feature_names
        
        # Calculate plot dimensions based on number of features
        num_features = len(plot_features) if feature_names else len(self.feature_names)
        plt.figure(figsize=(10, num_features * plot_height))
        
        # Create the decision plot
        shap.decision_plot(
            base_value=shap_values.base_values[0],
            shap_values=shap_values.values,
            features=X_processed.values,
            feature_names=plot_features,
            feature_order=feature_order,
            # feature_idx=feature_mask,
            link="identity",
            legend_labels=[f'Sample {i}' for i in range(10)],
            highlight=0,  # Highlight the first sample
            show=False,
            # fontsize=10
        )
        
        # Add a title
        sample_desc = f"{len(sample_indices)} samples" if len(sample_indices) > 3 else f"samples {sample_indices}"
        plt.title(f"SHAP Decision Plot for {sample_desc}", fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_waterfall(self, X: pd.DataFrame, sample_idx: int = 0, max_display: int = 15):
        """
        Create a waterfall plot for a specific sample.
        
        Args:
            X: Input features
            sample_idx: Index of the sample to explain
            max_display: Maximum number of features to display
        """
        # Get a single sample
        X_single = X.iloc[[sample_idx]]
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X_single)
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[0], max_display=max_display)
        plt.tight_layout()
    
    def analyze_false_positives(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], 
                                max_samples: int = 10) -> Dict:
        """
        Analyze false positives using SHAP values.
        """
        return self.analyze_misclassifications(X, y, max_samples, 'false_positives')
    
    def analyze_false_negatives(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], 
                                max_samples: int = 10) -> Dict:
        """
        Analyze false negatives using SHAP values.
        """
        return self.analyze_misclassifications(X, y, max_samples, 'false_negatives')
    
    def analyze_misclassifications(self, X: pd.DataFrame, 
                                   y: Union[pd.Series, np.ndarray], 
                                   max_samples: int = 10,
                                   fp_or_fn: str = 'both') -> Dict:
        """
        Analyze false positives and false negatives using SHAP values.
        
        Args:
            X: Test features
            y: True labels
            max_samples: Maximum number of samples to analyze for each error type
            
        Returns:
            Dictionary with analysis results
        """
        # Get probability predictions instead of hard predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert to hard predictions
        
        # Store original probabilities for verification
        probabilities = y_pred_proba
        
        # Identify false positives and false negatives
        false_positives = (y_pred == 1) & (y == 0)
        false_negatives = (y_pred == 0) & (y == 1)
        
        # Get indices
        fp_indices = np.where(false_positives)[0]
        fn_indices = np.where(false_negatives)[0]
        
        # Also get the datetime indices (only addition to original function)
        fp_datetime_indices = X.index[fp_indices].tolist() if len(fp_indices) > 0 else []
        fn_datetime_indices = X.index[fn_indices].tolist() if len(fn_indices) > 0 else []
        
        print(f"Found {len(fp_indices)} false positives and {len(fn_indices)} false negatives")
        if len(fp_indices) > 0:
            print(f"False positive probabilities range: {probabilities[false_positives].min():.3f} - {probabilities[false_positives].max():.3f}")
        
        results = {}
        
        # Analyze false positives
        if len(fp_indices) > 0 and (fp_or_fn == 'false_positives' or fp_or_fn == 'both'):
            print("\n===== FALSE POSITIVES ANALYSIS =====")
            print("Cases where the model predicted clear sky (1) but actual label was not clear (0)")
            
            # Get false positive samples
            X_fp = X.iloc[fp_indices]
            
            # Compute SHAP values
            fp_shap_values = self.compute_shap_values(X_fp)
            
            # Plot summary
            plt.figure(figsize=(12, 8))
            plt.title("SHAP Summary for False Positives")
            shap.summary_plot(fp_shap_values.values, self._prepare_data(X_fp), 
                             max_display=15, show=False)
            plt.tight_layout()
            plt.show()
            
            # Store results
            results['false_positives'] = {
                'indices': fp_indices,
                'datetime_indices': fp_datetime_indices, 
                'shap_values': fp_shap_values
            }
            
            # Show individual explanations for a few false positives
            for i in range(min(3, len(fp_indices))):
                idx = fp_indices[i]
                print(f"\nFalse Positive Example #{i+1} (Index: {idx})")
                
                # Display key feature values
                self._print_key_features(X.iloc[idx])
                
                title = f"SHAP Force Plot for False Positive #{i+1}"
                self._plot_force_with_rounded_values(fp_shap_values, i, title, y_true=y[idx])
                # Plot force plot
                # plt.figure(figsize=(20, 3))
                # shap.plots.force(fp_shap_values[i], 
                #                  show=False, 
                #                  matplotlib=True,
                #                  text_rotation=45)
                # plt.title(f"SHAP Force Plot for False Positive #{i+1}", fontsize=16, pad=20, y=2.0)
                # plt.tight_layout()
                # plt.show()
        
        # Analyze false negatives
        if len(fn_indices) > 0 and (fp_or_fn == 'false_negatives' or fp_or_fn == 'both'):
            print("\n===== FALSE NEGATIVES ANALYSIS =====")
            print("Cases where the model predicted not clear sky (0) but actual label was clear (1)")
            
            # Get false negative samples
            X_fn = X.iloc[fn_indices]
            
            # Compute SHAP values
            fn_shap_values = self.compute_shap_values(X_fn)
            
            # Plot summary
            plt.figure(figsize=(12, 8))
            plt.title("SHAP Summary for False Negatives")
            shap.summary_plot(fn_shap_values.values, self._prepare_data(X_fn), 
                             max_display=15, show=False)
            plt.tight_layout()
            plt.show()
            
            # Store results
            results['false_negatives'] = {
                'indices': fn_indices,
                'datetime_indices': fn_datetime_indices, 
                'shap_values': fn_shap_values
            }
            
            # Show individual explanations for a few false negatives
            for i in range(min(3, len(fn_indices))):
                idx = fn_indices[i]
                print(f"\nFalse Negative Example #{i+1} (Index: {idx})")
                
                # Display key feature values
                self._print_key_features(X.iloc[idx])
                
                title = f"SHAP Force Plot for False Negative #{i+1}"
                # Plot force plot
                self._plot_force_with_rounded_values(fn_shap_values, i, title, y_true=y[idx])
        
        return results
    
    def _print_key_features(self, sample: pd.Series):
        """
        Print key feature values for a sample.
        
        Args:
            sample: Sample data (Series)
        """
        print("Key feature values:")
        key_features = ['direct_normal', 'global_horizontal', 'diffuse_horizontal', 
                       'kt', 'kb', 'kd', 'clear_sky_score', 'zenith', 'hour']
        
        # Filter to only include features that exist in the sample
        available_features = [f for f in key_features if f in sample.index]
        
        for feature in available_features:
            value = sample[feature]
            if isinstance(value, (int, float)):
                print(f"  {feature}: {value:.4f}")
            else:
                print(f"  {feature}: {value}")
    
    def analyze_temporal_patterns(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        """
        Analyze when misclassifications occur in time (hour of day, month, etc.)
        
        Args:
            X: Test features
            y: True labels
        """
        # Get predictions
        X = self.model._prepare_features(X, is_training=False)
        y_pred = self.model.predict(X)
        
        # Create a DataFrame with predictions and actual values
        results_df = pd.DataFrame({
            'actual': y,
            'predicted': y_pred,
            'correct': y == y_pred,
            'false_positive': (y_pred == 1) & (y == 0),
            'false_negative': (y_pred == 0) & (y == 1)
        })
        
        # Add temporal information if available
        if isinstance(X.index, pd.DatetimeIndex):
            results_df['hour'] = X.index.hour
            results_df['month'] = X.index.month
            results_df['day_of_year'] = X.index.dayofyear
            
            # Plot misclassifications by hour of day
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            hourly_fp = results_df.groupby('hour')['false_positive'].mean() * 100
            hourly_fn = results_df.groupby('hour')['false_negative'].mean() * 100
            hourly_fp.plot(kind='bar', color='red', alpha=0.7, label='False Positives')
            hourly_fn.plot(kind='bar', color='blue', alpha=0.7, label='False Negatives')
            plt.title('Misclassification Rate by Hour of Day')
            plt.xlabel('Hour')
            plt.ylabel('Error Rate (%)')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            plt.subplot(1, 2, 2)
            monthly_fp = results_df.groupby('month')['false_positive'].mean() * 100
            monthly_fn = results_df.groupby('month')['false_negative'].mean() * 100
            monthly_fp.plot(kind='bar', color='red', alpha=0.7, label='False Positives')
            monthly_fn.plot(kind='bar', color='blue', alpha=0.7, label='False Negatives')
            plt.title('Misclassification Rate by Month')
            plt.xlabel('Month')
            plt.ylabel('Error Rate (%)')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        else:
            print("No datetime index found in the data. Cannot analyze temporal patterns.")
        
        return results_df
    
    def analyze_feature_thresholds(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], 
                                top_n_features: int = 5):
        """
        Analyze feature thresholds that lead to misclassifications.
        
        Args:
            X: Test features
            y: True labels
            top_n_features: Number of top features to analyze
        """
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Compute SHAP values for all samples
        shap_values = self.compute_shap_values(X, sample_size=min(1000, len(X)))  # Limit to 1000 samples for performance
        
        # Get the top N most important features
        feature_importance = np.abs(shap_values.values).mean(0)
        top_features_idx = np.argsort(-feature_importance)[:top_n_features]
        top_features = [self.feature_names[i] for i in top_features_idx]
        
        print(f"Analyzing top {top_n_features} features: {', '.join(top_features)}")
        
        # Create a figure with subplots for each top feature
        fig, axes = plt.subplots(len(top_features), 1, figsize=(12, 4 * len(top_features)))
        
        # If only one feature, make axes iterable
        if len(top_features) == 1:
            axes = [axes]
        
        # For each top feature, plot its distribution for correct and incorrect predictions
        for i, feature in enumerate(top_features):
            # Get the feature values from the processed data
            feature_idx = self.feature_names.index(feature)
            feature_values = X_processed.iloc[:, feature_idx]
            
            # Print some diagnostic information
            print(f"\nFeature: {feature}")
            print(f"  Min: {feature_values.min()}, Max: {feature_values.max()}")
            print(f"  Correct predictions: {sum(y == y_pred)}, Incorrect: {sum(y != y_pred)}")
            print(f"  False positives: {sum((y_pred == 1) & (y == 0))}, False negatives: {sum((y_pred == 0) & (y == 1))}")
            
            # Create a DataFrame with the feature and prediction correctness
            plot_df = pd.DataFrame({
                'feature_value': feature_values,
                'correct': y == y_pred,
                'false_positive': (y_pred == 1) & (y == 0),
                'false_negative': (y_pred == 0) & (y == 1)
            })
            
            # Check if we have data to plot
            if sum(plot_df['correct']) > 0:
                axes[i].hist(plot_df[plot_df['correct']]['feature_value'], 
                        bins=30, alpha=0.5, label=f'Correct ({sum(plot_df["correct"])})', color='green')
            
            if sum(plot_df['false_positive']) > 0:
                axes[i].hist(plot_df[plot_df['false_positive']]['feature_value'], 
                        bins=30, alpha=0.5, label=f'False Positive ({sum(plot_df["false_positive"])})', color='red')
            
            if sum(plot_df['false_negative']) > 0:
                axes[i].hist(plot_df[plot_df['false_negative']]['feature_value'], 
                        bins=30, alpha=0.5, label=f'False Negative ({sum(plot_df["false_negative"])})', color='blue')
            
            axes[i].set_title(f'Distribution of {feature} by Prediction Outcome')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Count')
            axes[i].legend()
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0, 
                        top_features: int = 10, include_force_plot: bool = True):
        """
        Provide a comprehensive explanation for a prediction.
        
        Args:
            X: Input features (DataFrame or Series)
            sample_idx: Index of the sample to explain (only used if X is a DataFrame)
            top_features: Number of top features to show
            include_force_plot: Whether to include a force plot
        """

        X_single = X.iloc[[sample_idx]]
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X_single)
        
        # Prepare the data for plotting
        X_processed = self.model._prepare_features(X_single, is_training=False)

        # Get prediction
        prediction = self.model.predict_proba(X_single)[0, 1]
        predicted_class = "Clear Sky" if prediction >= 0.5 else "Not Clear Sky"
        
        # Print basic information
        print(f"Sample: {X_single.index[0]}")
        print(f"Prediction: {predicted_class} (probability: {prediction:.4f})")
        print("\nTop contributing features:")
        
        # Get feature contributions
        feature_values = X_processed.iloc[0]
        shap_values_array = shap_values.values[0]
        
        # Sort features by absolute SHAP value
        indices = np.argsort(np.abs(shap_values_array))[::-1]
        
        # Display top features
        for i in range(min(top_features, len(indices))):
            idx = indices[i]
            feature_name = self.feature_names[idx]
            feature_value = feature_values[idx]
            shap_value = shap_values_array[idx]
            direction = "increases" if shap_value > 0 else "decreases"
            
            print(f"{i+1}. {feature_name} = {feature_value:.4f} ({direction} probability by {abs(shap_value):.4f})")
        
        # Include force plot if requested
        if include_force_plot:
            # Create a force plot with the same approach that worked for plot_force
            title = f"SHAP Force Plot for {predicted_class} Prediction"
            # self._plot_force_with_rounded_values(shap_values, 0, title, y_true=y[0])
            # plt.figure(figsize=(20, 7))
            # shap.plots.force(
            #     base_value=shap_values.base_values[0],
            #     shap_values=shap_values.values[0],
            #     features=feature_values.values,
            #     feature_names=self.feature_names,
            #     matplotlib=True,
            #     show=False,
            # )
            # plt.title(f"SHAP Force Plot for {predicted_class} Prediction", fontsize=16, pad=20, y=2.0)
            # plt.tight_layout()
            # plt.show()

    def plot_feature_interaction(self, X: pd.DataFrame, feature1: Union[str, int], 
                                feature2: Union[str, int], highlight_misclassifications=True,
                                sample_size: Optional[int] = 1000):
        """
        Create an enhanced plot showing the interaction between two features.
        
        Args:
            X (pd.DataFrame): Data to use for the plot
            feature1 (str or int): Name or index of the first feature
            feature2 (str or int): Name or index of the second feature
            highlight_misclassifications (bool): Whether to highlight misclassifications
            sample_size (int, optional): Number of samples to use (for performance). If None, use all.
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Convert feature indices to names if needed
        if isinstance(feature1, int):
            feature1 = X.columns[feature1]
        if isinstance(feature2, int):
            feature2 = X.columns[feature2]
        
        # Compute SHAP values
        shap_values = self.compute_shap_values(X, sample_size)
        
        # Get the data used for the plot
        # Since we're sampling from X in compute_shap_values, we need to use the same subset
        if len(X) > sample_size:
            # If we sampled, use the first sample_size rows
            X_subset = X.iloc[:sample_size]
        else:
            # If we didn't sample, use all of X
            X_subset = X
        
        # Prepare the data for plotting
        X_processed = self._prepare_data(X_subset)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get data for plotting
        x = X_processed[feature1]
        y = X_processed[feature2]
        
        # Calculate combined SHAP impact
        combined_shap = shap_values[:, feature1].values + shap_values[:, feature2].values
        
        # Create scatter plot
        scatter = ax.scatter(x, y, c=combined_shap, cmap='coolwarm', alpha=0.7, s=50)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f"Combined SHAP Impact ({feature1} + {feature2})", fontsize=12)
        
        # Add reference lines and regions based on features
        if (feature1 == "direct_to_diffuse_ratio" and feature2 == "kt") or \
        (feature2 == "direct_to_diffuse_ratio" and feature1 == "kt"):
            
            # Determine which axis is which feature
            if feature1 == "direct_to_diffuse_ratio":
                ratio_axis = ax.xaxis
                kt_axis = ax.yaxis
                ratio_data = x
                kt_data = y
            else:
                ratio_axis = ax.yaxis
                kt_axis = ax.xaxis
                ratio_data = y
                kt_data = x
            
            # Add reference line for kt = 1.0 (cloud enhancement threshold)
            if feature1 == "kt":
                ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, 
                        label='Cloud enhancement threshold (kt=1.0)')
                # Highlight cloud enhancement region
                ax.axvspan(1.0, ax.get_xlim()[1], alpha=0.1, color='red', 
                        label='Cloud enhancement region')
            else:
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                        label='Cloud enhancement threshold (kt=1.0)')
                # Highlight cloud enhancement region
                ax.axhspan(1.0, ax.get_ylim()[1], alpha=0.1, color='red', 
                        label='Cloud enhancement region')
            
            # Add reference line for direct_to_diffuse_ratio
            # Use a more appropriate threshold based on data distribution
            ratio_threshold = 0.1  # Adjusted from 0.05 based on data distribution
            if feature1 == "direct_to_diffuse_ratio":
                ax.axvline(x=ratio_threshold, color='blue', linestyle='--', alpha=0.7, 
                        label=f'Low direct_to_diffuse_ratio threshold ({ratio_threshold})')
            else:
                ax.axhline(y=ratio_threshold, color='blue', linestyle='--', alpha=0.7, 
                        label=f'Low direct_to_diffuse_ratio threshold ({ratio_threshold})')
        
        # Add other common reference lines for specific features
        elif feature1 == "kt" or feature2 == "kt":
            kt_feature = feature1 if feature1 == "kt" else feature2
            
            if kt_feature == feature1:
                ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, 
                        label='Cloud enhancement threshold (kt=1.0)')
            else:
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                        label='Cloud enhancement threshold (kt=1.0)')
        
        elif feature1 == "direct_to_diffuse_ratio" or feature2 == "direct_to_diffuse_ratio":
            ratio_feature = feature1 if feature1 == "direct_to_diffuse_ratio" else feature2
            ratio_threshold = 0.1  # Adjusted threshold
            
            if ratio_feature == feature1:
                ax.axvline(x=ratio_threshold, color='blue', linestyle='--', alpha=0.7, 
                        label=f'Low direct_to_diffuse_ratio threshold ({ratio_threshold})')
            else:
                ax.axhline(y=ratio_threshold, color='blue', linestyle='--', alpha=0.7, 
                        label=f'Low direct_to_diffuse_ratio threshold ({ratio_threshold})')
        
        # Highlight misclassifications if requested
        if highlight_misclassifications and hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            # Get predictions if not already computed
            if not hasattr(self, 'y_pred_test'):
                self.y_pred_test = (self.model.predict_proba(self.X_test)[:, 1] >= 0.5).astype(int)
            
            # Find which samples in X_subset are also in X_test
            common_indices = []
            for i, row in X_subset.iterrows():
                # Find matching rows in X_test
                matches = ((self.X_test == row).all(axis=1))
                if matches.any():
                    test_idx = matches.idxmax()
                    common_indices.append((i, test_idx))
            
            # Highlight false positives
            fp_pairs = [(i, test_idx) for i, test_idx in common_indices 
                    if self.y_pred_test[test_idx] == 1 and self.y_test[test_idx] == 0]
            
            if fp_pairs:
                fp_sample_indices = [i for i, _ in fp_pairs]
                ax.scatter(
                    X_processed.iloc[fp_sample_indices][feature1],
                    X_processed.iloc[fp_sample_indices][feature2],
                    s=100, edgecolor='black', facecolor='none', linewidth=2,
                    label='False Positives'
                )
            
            # Highlight false negatives
            fn_pairs = [(i, test_idx) for i, test_idx in common_indices 
                    if self.y_pred_test[test_idx] == 0 and self.y_test[test_idx] == 1]
            
            if fn_pairs:
                fn_sample_indices = [i for i, _ in fn_pairs]
                ax.scatter(
                    X_processed.iloc[fn_sample_indices][feature1],
                    X_processed.iloc[fn_sample_indices][feature2],
                    s=100, edgecolor='blue', facecolor='none', linewidth=2,
                    label='False Negatives'
                )
        
        # Add data quality warning if negative ratios are present
        if ((feature1 == "direct_to_diffuse_ratio" and (X_processed[feature1] < 0).any()) or 
            (feature2 == "direct_to_diffuse_ratio" and (X_processed[feature2] < 0).any())):
            plt.figtext(0.5, 0.01, 
                    "Warning: Dataset contains negative direct-to-diffuse ratios, which may indicate data quality issues.",
                    ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Add title and labels
        feature1_display = feature1.replace('_', ' ').title()
        feature2_display = feature2.replace('_', ' ').title()
        ax.set_title(f"Interaction Between {feature1_display} and {feature2_display}", fontsize=16)
        ax.set_xlabel(feature1_display, fontsize=14)
        ax.set_ylabel(feature2_display, fontsize=14)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add legend if we have any labeled elements
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend(fontsize=12)
        
        plt.tight_layout()
        
        return fig

    def analyze_false_positives(self, X: pd.DataFrame, max_samples=3, figsize=(15, 15), sample_size=1000):
        """
        Create a comprehensive analysis of false positives.
        
        Args:
            max_samples (int): Maximum number of individual false positives to analyze
            figsize (tuple): Figure size
            sample_size (int): Number of samples to use for SHAP calculations
            
        Returns:
            matplotlib.figure.Figure: The created figure with multiple subplots
        """
        # Get predictions if not already computed
        if not hasattr(self, 'y_pred_test'):
            # 
            self.y_pred_test = (self.model.predict_proba(self.X_test)[:, 1] >= 0.5).astype(int)
        
        # Identify false positives
        fp_indices = np.where((self.y_pred_test == 1) & (self.y_test == 0))[0]
        
        if len(fp_indices) == 0:
            print("No false positives found in the test set.")
            return None
        
        # Get SHAP values for all test data
        if not hasattr(self, 'shap_values_test') or self.shap_values_test is None:
            print("Computing SHAP values for test data...")
            self.shap_values_test = self.compute_shap_values(self.X_test, min(len(self.X_test), sample_size))
        
        # Get SHAP values specifically for false positives
        X_fp = self.X_test.iloc[fp_indices]
        shap_values_fp = self.compute_shap_values(X_fp, min(len(X_fp), sample_size))
        
        # Calculate number of rows needed
        n_rows = 3 + min(max_samples, len(fp_indices))  # Summary plots + feature interactions + individual samples
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=figsize)
        
        # Define grid layout
        gs = gridspec.GridSpec(n_rows, 2, figure=fig)
        
        # Plot 1: SHAP summary for false positives
        ax1 = fig.add_subplot(gs[0, 0])
        shap.plots.beeswarm(shap_values_fp, show=False, max_display=15)
        ax1.set_title("SHAP Summary for False Positives", fontsize=14)
        
        # Plot 2: Feature importance for false positives
        ax2 = fig.add_subplot(gs[0, 1])
        shap.plots.bar(shap_values_fp.abs.mean(0), show=False, max_display=15)
        ax2.set_title("Feature Importance for False Positives", fontsize=14)
        
        # Plot 3: Scatter plot of direct_to_diffuse_ratio vs kt for false positives
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Prepare data for scatter plot
        X_fp_processed = self._prepare_data(X_fp)
        
        # Create scatter plot
        scatter = ax3.scatter(
            X_fp_processed["direct_to_diffuse_ratio"],
            X_fp_processed["kt"],
            c=shap_values_fp[:, "direct_to_diffuse_ratio"].values + shap_values_fp[:, "kt"].values,
            cmap='coolwarm',
            alpha=0.7,
            s=50
        )
        
        # Add reference lines
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                label='Cloud enhancement threshold (kt=1.0)')
        ax3.axvline(x=0.1, color='blue', linestyle='--', alpha=0.7, 
                label='Low direct_to_diffuse_ratio threshold (0.1)')
        
        # Highlight cloud enhancement region
        ax3.axhspan(1.0, ax3.get_ylim()[1], alpha=0.1, color='red', 
                label='Cloud enhancement region')
        
        # Add labels and title
        ax3.set_title("False Positives: Direct-to-Diffuse Ratio vs. Kt", fontsize=14)
        ax3.set_xlabel("Direct To Diffuse Ratio", fontsize=12)
        ax3.set_ylabel("Kt", fontsize=12)
        ax3.legend(fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label("Combined SHAP Impact", fontsize=10)
        
        # Add data quality warning if needed
        if (X_fp_processed["direct_to_diffuse_ratio"] < 0).any():
            ax3.text(0.5, -0.15, 
                    "Warning: Dataset contains negative direct-to-diffuse ratios, which may indicate data quality issues.",
                    ha="center", transform=ax3.transAxes, fontsize=10, 
                    bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # Plot 4: Distribution of global horizontal for false positives
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Create histogram
        bins = np.linspace(0, max(X_fp_processed["global_horizontal"]) * 1.1, 30)
        ax4.hist(X_fp_processed["global_horizontal"], bins=bins, color="red", alpha=0.7)
        
        # Add reference line for low irradiance
        low_irradiance_threshold = 50  # W/m²
        ax4.axvline(x=low_irradiance_threshold, color='blue', linestyle='--', alpha=0.7,
                label=f'Low irradiance threshold ({low_irradiance_threshold} W/m²)')
        
        # Calculate percentage of false positives with low irradiance
        low_irradiance_pct = (X_fp_processed["global_horizontal"] < low_irradiance_threshold).mean() * 100
        
        # Add annotation
        ax4.text(0.95, 0.95, f"{low_irradiance_pct:.1f}% of false positives\nhave GHI < {low_irradiance_threshold} W/m²",
                transform=ax4.transAxes, ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Add labels and title
        ax4.set_title("Distribution of Global Horizontal for False Positives", fontsize=14)
        ax4.set_xlabel("Global Horizontal Irradiance (W/m²)", fontsize=12)
        ax4.set_ylabel("Count", fontsize=12)
        ax4.legend(fontsize=10)
        
        # Plot 5: Cloud enhancement analysis
        ax5 = fig.add_subplot(gs[2, 0])
        
        # Calculate percentage of false positives with cloud enhancement
        cloud_enhanced = (X_fp_processed["kt"] > 1.0)
        cloud_enhanced_pct = cloud_enhanced.mean() * 100
        
        # Create pie chart
        labels = ['Cloud Enhancement (kt > 1.0)', 'No Cloud Enhancement (kt ≤ 1.0)']
        sizes = [cloud_enhanced_pct, 100 - cloud_enhanced_pct]
        colors = ['red', 'lightgray']
        explode = (0.1, 0)  # explode the 1st slice
        
        ax5.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
        ax5.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax5.set_title("Percentage of False Positives with Cloud Enhancement", fontsize=14)
        
        # Plot 6: Direct-to-diffuse ratio analysis
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Define ratio categories
        ratio_bins = [-np.inf, 0, 0.1, 0.5, 1.0, np.inf]
        ratio_labels = ['Negative (invalid)', 'Very low (0-0.1)', 'Low (0.1-0.5)', 'Medium (0.5-1.0)', 'High (>1.0)']
        
        # Categorize the ratios
        ratio_categories = pd.cut(X_fp_processed["direct_to_diffuse_ratio"], bins=ratio_bins, labels=ratio_labels)
        ratio_counts = ratio_categories.value_counts().sort_index()
        
        # Create bar chart
        bars = ax6.bar(ratio_labels, ratio_counts, color='skyblue')
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height/len(X_fp_processed)*100:.1f}%',
                    ha='center', va='bottom', fontsize=9)
        
        # Add labels and title
        ax6.set_title("Distribution of Direct-to-Diffuse Ratio in False Positives", fontsize=14)
        ax6.set_xlabel("Direct-to-Diffuse Ratio Category", fontsize=12)
        ax6.set_ylabel("Count", fontsize=12)
        ax6.set_xticklabels(ratio_labels, rotation=45, ha='right')
        
        # Individual false positive analysis
        for i in range(min(max_samples, len(fp_indices))):
            idx = fp_indices[i]
            
            # Force plot for individual false positive
            ax_force = fig.add_subplot(gs[i+3, 0])
            shap.plots.force(self.shap_values_test[idx], show=False, matplotlib=True)
            ax_force.set_title(f"SHAP Force Plot for False Positive #{i+1}", fontsize=14)
            
            # Key metrics for this false positive
            ax_metrics = fig.add_subplot(gs[i+3, 1])
            
            # Extract key metrics
            metrics = {
                'direct_to_diffuse_ratio': self.X_test.iloc[idx]['direct_to_diffuse_ratio'],
                'kt': self.X_test.iloc[idx]['kt'],
                'global_horizontal': self.X_test.iloc[idx]['global_horizontal'],
                'diffuse_horizontal': self.X_test.iloc[idx]['diffuse_horizontal'],
                'kb_to_kd_ratio': self.X_test.iloc[idx]['kb_to_kd_ratio'],
                'azimuth': self.X_test.iloc[idx]['azimuth'],
                'zenith': self.X_test.iloc[idx]['zenith'],
                'model_output': self.model.predict_proba(self.X_test.iloc[idx:idx+1])[0, 1]
            }
            
            # Create a horizontal bar chart of key metrics
            y_pos = np.arange(len(metrics))
            metric_values = list(metrics.values())
            metric_names = list(metrics.keys())
            
            bars = ax_metrics.barh(y_pos, metric_values, align='center')
            ax_metrics.set_yticks(y_pos)
            ax_metrics.set_yticklabels(metric_names)
            ax_metrics.set_title(f"Key Metrics for False Positive #{i+1}", fontsize=14)
            
            # Add value labels to the bars
            for bar in bars:
                width = bar.get_width()
                ax_metrics.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2f}', ha='left', va='center')
            
            # Add annotations about cloud enhancement and direct-to-diffuse ratio
            has_cloud_enhancement = metrics['kt'] > 1.0
            has_low_ratio = metrics['direct_to_diffuse_ratio'] < 0.1
            
            annotation_text = []
            if has_cloud_enhancement:
                annotation_text.append("• Shows cloud enhancement (kt > 1.0)")
            if has_low_ratio:
                annotation_text.append("• Has very low direct-to-diffuse ratio (< 0.1)")
            if metrics['global_horizontal'] < 50:
                annotation_text.append("• Low global irradiance (< 50 W/m²)")
            
            if annotation_text:
                ax_metrics.text(0.5, -0.15, "\n".join(annotation_text),
                            transform=ax_metrics.transAxes, ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
        
        # Add overall summary at the bottom
        plt.figtext(0.5, 0.02, 
                f"Summary: Analyzed {len(fp_indices)} false positives. " +
                f"{cloud_enhanced_pct:.1f}% show cloud enhancement (kt > 1.0). " +
                f"{low_irradiance_pct:.1f}% have low irradiance (< 50 W/m²).",
                ha="center", fontsize=14, 
                bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        
        return fig

    def _plot_force_with_rounded_values(self, shap_values, idx=0, title="SHAP Force Plot", y_true=0, figsize=(20, 3)):
        """
        Plot a SHAP force plot with rounded feature values to avoid overlapping labels.
        """
        import copy
        exp = copy.deepcopy(shap_values[idx])
        
        # Round the feature values
        if hasattr(exp, 'data') and exp.data is not None:
            if isinstance(exp.data, np.ndarray):
                for i in range(len(exp.data)):
                    if isinstance(exp.data[i], (int, float)):
                        exp.data[i] = round(float(exp.data[i]), 2)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Use force_plot without logit link since values are already in log-odds space
        shap.plots.force(
            exp,
            show=False,
            matplotlib=True,
            text_rotation=30 
        )
        
        # Add probability scale on top
        base_value = exp.base_values if isinstance(exp.base_values, float) else exp.base_values[0]
        total_value = base_value + np.sum(exp.values)
        prob = 1 / (1 + np.exp(-total_value))  # Convert to probability
        
        pred = 1 if prob >= 0.5 else 0
    
        if y_true == 0:  # Actually not clear sky
            if pred == 1:
                classification = "FALSE POSITIVE (Incorrectly predicted as clear sky)"
            else:
                classification = "TRUE NEGATIVE (Correctly predicted as not clear sky)"
        else:  # Actually clear sky
            if pred == 1:
                classification = "TRUE POSITIVE (Correctly predicted as clear sky)"
            else:
                classification = "FALSE NEGATIVE (Incorrectly predicted as not clear sky)"
        
        plt.title(f"{title}\nPrediction Probability: {prob:.3f}\nClassification: {classification}", 
            fontsize=16, pad=20, y=2.0)
        plt.tight_layout()
        # plt.show()


    def _generate_text_explanation(self, sample: pd.Series, prediction: int, 
                                 probability: float, shap_values: Any):
        """
        Generate a textual explanation for a prediction.
        
        Args:
            sample: Sample data
            prediction: Model prediction (0 or 1)
            probability: Prediction probability
            shap_values: SHAP values for the sample
        """
        # Get feature values and their SHAP values
        feature_values = {}
        for feature in self.feature_names:
            if feature in sample.index:
                feature_values[feature] = sample[feature]
        
        # Get the top contributing features (positive and negative)
        feature_impacts = list(zip(self.feature_names, shap_values.values))
        sorted_impacts = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)
        
        # Get top 5 contributing features
        top_features = sorted_impacts[:5]
        
        # Generate explanation
        explanation = []
        explanation.append(f"The model {'classified' if prediction == 1 else 'did not classify'} this as a clear sky period with {probability:.1%} confidence.")
        
        # Add explanation about key contributing features
        explanation.append("\nThe main factors influencing this decision were:")
        
        for feature, impact in top_features:
            if impact > 0:
                direction = "increased"
                effect = "more likely to be clear sky"
            else:
                direction = "decreased"
                effect = "less likely to be clear sky"
            
            if feature in feature_values:
                value = feature_values[feature]
                if isinstance(value, (int, float)):
                    explanation.append(f"- {feature} = {value:.4f} ({direction} the probability, making it {effect})")
                else:
                    explanation.append(f"- {feature} = {value} ({direction} the probability, making it {effect})")
        
        # Print the explanation
        print("\nEXPLANATION:")
        print("\n".join(explanation))


    def cluster_misclassifications(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], 
                                n_clusters: int = 3, misclassification_type: str = 'both',
                                feature_subset: Optional[List[str]] = None,
                                use_shap_values: bool = True, 
                                dim_reduction_method: str = 'pca',  # Add this parameter
                                umap_n_neighbors: int = 30,         # UMAP specific parameters
                                umap_min_dist: float = 0.1,
                                random_state: int = 41):
        """
        Cluster false positives and false negatives to identify patterns.
        
        Args:
            X: Test features
            y: True labels
            n_clusters: Number of clusters to create
            misclassification_type: 'false_positives', 'false_negatives', or 'both'
            feature_subset: Optional list of features to use for clustering (if None, uses all)
            use_shap_values: Whether to cluster based on SHAP values (True) or raw features (False)
            dim_reduction_method: Dimensionality reduction method before clustering ('pca', 'umap', or 'both')
            umap_n_neighbors: Number of neighbors for UMAP (higher values = more global structure)
            umap_min_dist: Minimum distance for UMAP (lower values = tighter clusters)
            random_state: Random seed for clustering algorithms
            
        Returns:
            Dictionary with clustering results
        """
        # Validate dim_reduction_method parameter
        if dim_reduction_method not in ['pca', 'umap', 'both']:
            raise ValueError("dim_reduction_method must be one of: 'pca', 'umap', 'both'")
        
        # Check for UMAP dependency
        if dim_reduction_method in ['umap', 'both']:
            try:
                import umap
            except ImportError:
                raise ImportError(
                    "UMAP is required for UMAP dimensionality reduction. "
                    "Install it with 'pip install umap-learn'"
                )

        # Process the data with the existing methods
        X_processed = self._prepare_features(X, is_training=False)
        
        # Get predictions using processed data
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Identify misclassifications
        false_positives = (y_pred == 1) & (y == 0)
        false_negatives = (y_pred == 0) & (y == 1)
        
        print("\nDebug Information:")
        print(f"Total samples: {len(X)}")
        print(f"Predicted positives (≥0.5): {sum(y_pred == 1)}")
        print(f"Actual positives(y=1): {sum(y == 1)}")
        print(f"False positives: {sum(false_positives)}")
        print(f"False negatives: {sum(false_negatives)}")
        print(f"False positive probabilities range: {y_pred_proba[false_positives].min():.3f} - {y_pred_proba[false_positives].max():.3f}")
        
        # Display sample misclassifications for debugging
        fp_indices = np.where(false_positives)[0]
        print("\nDEBUG - False Positive Verification:")
        print(f"First 5 identified false positives:")
        for idx in fp_indices[:5]:
            print(f"\nIndex {idx}:")
            print(f"Predicted probability: {y_pred_proba[idx]:.3f}")
            print(f"Actual label (y): {y[idx]}")
            print(f"Predicted label (y_pred): {y_pred[idx]}")
            print(f"Is false positive: {y_pred[idx] == 1 and y[idx] == 0}")
        
        results = {}
        
        # Process false positives
        if misclassification_type in ['false_positives', 'both'] and false_positives.sum() > 0:
            if dim_reduction_method == 'both':
                # Use both PCA and UMAP
                results['false_positives_pca'] = self._perform_clustering(
                    X, y, false_positives, 'false_positives', X_processed, 
                    n_clusters, feature_subset, use_shap_values, 'pca',
                    umap_n_neighbors, umap_min_dist, random_state
                )
                results['false_positives_umap'] = self._perform_clustering(
                    X, y, false_positives, 'false_positives', X_processed, 
                    n_clusters, feature_subset, use_shap_values, 'umap',
                    umap_n_neighbors, umap_min_dist, random_state
                )
            else:
                # Use the selected method
                results['false_positives'] = self._perform_clustering(
                    X, y, false_positives, 'false_positives', X_processed, 
                    n_clusters, feature_subset, use_shap_values, dim_reduction_method,
                    umap_n_neighbors, umap_min_dist, random_state
                )
        
        # Process false negatives
        if misclassification_type in ['false_negatives', 'both'] and false_negatives.sum() > 0:
            if dim_reduction_method == 'both':
                # Use both PCA and UMAP
                results['false_negatives_pca'] = self._perform_clustering(
                    X, y, false_negatives, 'false_negatives', X_processed, 
                    n_clusters, feature_subset, use_shap_values, 'pca',
                    umap_n_neighbors, umap_min_dist, random_state
                )
                results['false_negatives_umap'] = self._perform_clustering(
                    X, y, false_negatives, 'false_negatives', X_processed, 
                    n_clusters, feature_subset, use_shap_values, 'umap',
                    umap_n_neighbors, umap_min_dist, random_state
                )
            else:
                # Use the selected method
                results['false_negatives'] = self._perform_clustering(
                    X, y, false_negatives, 'false_negatives', X_processed, 
                    n_clusters, feature_subset, use_shap_values, dim_reduction_method,
                    umap_n_neighbors, umap_min_dist, random_state
                )
        
        return results

    def clusters_with_decision_plot(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], 
                                cluster_results: Dict, dim_method: str, 
                                samples_per_cluster: int = 100,
                                feature_names: Optional[List[str]] = None,
                                plot_height: float = 0.5):
        """
        Create SHAP decision plots for samples from each cluster to analyze misclassification patterns.
        Uses an image-based approach to ensure reliable subplot layout.
        
        Args:
            X: Original feature dataset
            y: True labels
            cluster_results: Results from cluster_misclassifications method
            dim_method: Dimensionality reduction method ('pca', 'umap', or 'both')
            samples_per_cluster: Maximum number of samples to show per cluster
            feature_names: Names of features for the plot
            plot_height: Height of each decision plot
            
        Returns:
            Figure with decision plots comparing clustering methods
        """
        
        # Close any existing plots to avoid duplicates
        plt.close('all')
        
        # Determine which methods to visualize
        methods_to_vis = []
        if dim_method == 'both':
            methods_to_vis = ['umap', 'pca']
        else:
            methods_to_vis = [dim_method]
        
        # Store results for each method
        method_results = {}
        method_cluster_counts = {}
        
        # Extract data for each method
        for method in methods_to_vis:
            if f'false_positives_{method}' in cluster_results:
                method_results[method] = cluster_results[f'false_positives_{method}']
            elif 'false_positives' in cluster_results and cluster_results['false_positives'].get('dim_reduction_method') == method:
                method_results[method] = cluster_results['false_positives']
            else:
                print(f"Warning: No clustering results found for {method}")
                if method == dim_method:  # If this is the primary method requested, raise error
                    raise ValueError(f"No clustering results found for {method}")
                continue
            
            # Store cluster count for this method
            method_cluster_counts[method] = len(method_results[method]['cluster_stats'])
        
        # Determine the maximum number of clusters across methods
        max_clusters = max(method_cluster_counts.values()) if method_cluster_counts else 0
        if max_clusters == 0:
            raise ValueError("No clusters found for any method")
        
        # Determine grid layout
        if dim_method == 'both':
            n_rows = len(methods_to_vis)  # One row per method
            n_cols = max_clusters  # Columns = max number of clusters
        else:
            n_rows = 1  # Just one method
            n_cols = method_cluster_counts[dim_method]  # Number of clusters for that method
        
        # Create a temporary directory to store the individual plot images
        temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {temp_dir}")
        
        # Create a dictionary to store plot info and file paths
        plot_info = {}
        
        # Process each method and create individual decision plots
        for method in methods_to_vis:
            if method not in method_results:
                continue
                
            results = method_results[method]
            cluster_data = results['data']
            cluster_labels = results['cluster_labels']
            cluster_stats = results['cluster_stats']
            n_clusters = len(cluster_stats)
            
            # Get the datetime indices for these samples
            datetime_indices = results.get('datetime_indices', [])
            
            print(f"\n=== Processing {method.upper()} clusters ===")
            
            # Process each cluster for this method
            for cluster_id in range(n_clusters):
                # Generate a unique key for this plot
                plot_key = f"{method}_{cluster_id}"
                
                pattern = cluster_stats[cluster_id]['pattern']
                
                # Get all samples for this cluster
                cluster_mask = cluster_labels == cluster_id
                cluster_positions = np.where(cluster_mask)[0]
                
                # Map to datetime indices if available
                if len(datetime_indices) > 0 and len(cluster_positions) <= len(datetime_indices):
                    cluster_dt_indices = [datetime_indices[pos] for pos in cluster_positions]
                else:
                    # Fallback to using cluster_data index
                    cluster_samples = cluster_data[cluster_data['cluster'] == cluster_id]
                    cluster_dt_indices = cluster_samples.index.tolist()
                
                if len(cluster_dt_indices) == 0:
                    print(f"Cluster {cluster_id} has no samples")
                    plot_info[plot_key] = {
                        'status': 'empty',
                        'message': 'No samples found'
                    }
                    continue
                    
                print(f"{method.upper()} Cluster {cluster_id} ({pattern}) has {len(cluster_dt_indices)} samples")
                
                # Select a subset of representative samples
                if len(cluster_dt_indices) > samples_per_cluster:
                    # Simple random sampling for now
                    sampled_indices = np.random.choice(cluster_dt_indices, size=samples_per_cluster, replace=False)
                else:
                    sampled_indices = cluster_dt_indices
                
                # Find these indices in X
                valid_indices = [idx for idx in sampled_indices if idx in X.index]
                
                if len(valid_indices) == 0:
                    print(f"No valid indices found in X for {method} cluster {cluster_id}")
                    plot_info[plot_key] = {
                        'status': 'empty',
                        'message': 'No valid indices found'
                    }
                    continue
                
                print(f"Using {len(valid_indices)} samples for decision plot")
                
                # Create a copy to avoid modifying original
                X_selected = X.loc[valid_indices].copy()
                
                # Create measured_on column from index if needed
                if 'measured_on' not in X_selected.columns and isinstance(X_selected.index, pd.DatetimeIndex):
                    X_selected['measured_on'] = X_selected.index
                
                
                X_selected = X_selected.sort_values('measured_on')
                
            
                # Get key features for this cluster
                key_features = [
                    f for f, v in cluster_stats[cluster_id]['relative_diff'].items() 
                    if abs(v) > 25  # Features with >25% deviation from mean
                ]
                
                # Format key features with shorter names and more compact representation
                key_features_str = ", ".join([
                    f"{f.split('_')[0]}({cluster_stats[cluster_id]['relative_diff'][f]:.0f}%)" 
                    for f in key_features[:3] if f in cluster_stats[cluster_id]['relative_diff']
                ])
                
                # If no key features found, provide a default message
                if not key_features_str:
                    key_features_str = "No significant feature deviations"
                
                # Process data and create decision plot in a separate figure
                X_processed = self._prepare_data(X_selected)
                shap_values = self.explainer(X_processed)
                
                # Create a new figure for this decision plot with more space at bottom
                fig_decision = plt.figure(figsize=(7, 6.5))  # Add some space at bottom
                
                # Create the decision plot
                shap.decision_plot(
                    self.explainer.expected_value, 
                    shap_values.values, 
                    X_processed,
                    feature_names=feature_names or self.feature_names,
                    feature_order='importance',
                    plot_color='viridis',
                    alpha=0.5,
                    show=False,
                    # feature_display_range=15  # Fewer features for a more compact plot
                )
                
                # Add plot title with smaller font
                plt.title(f"{method.upper()} Cluster {cluster_id} ({pattern})", fontsize=11)
                
                # Find current axes
                ax = plt.gca()
                
                # Get the position of x-axis label to avoid overlap
                x_label_pos = ax.xaxis.get_label_position()
                
                # Move the x-axis label a bit higher to make room for key features
                ax.xaxis.set_label_coords(0.5, -0.1)  # Default is around -0.05
                
                # Add key features text as a separate text element with more space
                # Use figure coordinates instead of axes coordinates to avoid overlap
                plt.figtext(0.5, 0.02, f"Key features: {key_features_str}", 
                        ha='center', va='bottom', fontsize=9, 
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
                
                # Ensure the layout accommodates both the x-axis label and key features text
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space at bottom for key features
                
                # Save the figure to a file with high quality
                plot_file = os.path.join(temp_dir, f"{plot_key}.png")
                plt.savefig(plot_file, dpi=200, bbox_inches='tight')
                plt.close(fig_decision)
                
                # Store the plot info
                plot_info[plot_key] = {
                    'status': 'created',
                    'file': plot_file,
                    'method': method,
                    'cluster_id': cluster_id,
                    'pattern': pattern
                }
        
        # Create the main figure with subplots - use a single unified figure
        # Adjust heights - make rows closer together
        fig_height = 8 * n_rows  # Reduced height
        fig_width = 7 * n_cols   # Adjusted width
        
        # Create a new figure for the combined plots
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create GridSpec with reduced spacing between rows
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, 
                            wspace=0.2,   # Horizontal space between columns
                            hspace=0.1)   # Reduced vertical space between rows
        
        # Fill in the subplots grid with the saved images
        for row_idx, method in enumerate(methods_to_vis):
            if method not in method_results:
                continue
                
            for col_idx in range(method_cluster_counts[method]):
                plot_key = f"{method}_{col_idx}"
                ax = fig.add_subplot(gs[row_idx, col_idx])
                
                if plot_key in plot_info and plot_info[plot_key]['status'] == 'created':
                    # Load and display the saved image
                    img = plt.imread(plot_info[plot_key]['file'])
                    ax.imshow(img)
                    ax.axis('off')  # Hide axes
                elif plot_key in plot_info:
                    # Display the error or empty message
                    message = plot_info[plot_key].get('message', 'Unknown issue')
                    ax.text(0.5, 0.5, message, 
                            ha='center', va='center', fontsize=11,
                            transform=ax.transAxes)
                    ax.set_facecolor('#f8f8f8')  # Light gray background
                    ax.axis('off')
                else:
                    # Empty cell
                    ax.set_facecolor('#f8f8f8')
                    ax.axis('off')
        
        # Add overall title
        if dim_method == 'both':
            fig.suptitle(f"SHAP Decision Plots: UMAP vs PCA Clusters", fontsize=14, y=0.98)
        else:
            fig.suptitle(f"SHAP Decision Plots for {dim_method.upper()} Clusters", fontsize=14, y=0.98)
        
        # Adjust layout with tighter spacing
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
        
        # Clean up temporary files
        for info in plot_info.values():
            if info['status'] == 'created' and os.path.exists(info['file']):
                os.remove(info['file'])
               
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        # Only show this figure, not multiple figures
        plt.show()
        return fig


    def _select_representative_samples(self, X: pd.DataFrame, 
                                    cluster_indices: np.ndarray, 
                                    cluster_stats: Dict, 
                                    pattern: str, 
                                    n_samples: int) -> List[int]:
        """
        Select samples that best represent the cluster's pattern.
        
        Args:
            X: Original feature dataset
            cluster_indices: Indices of samples in the cluster
            cluster_stats: Statistics for the cluster
            pattern: Pattern name for the cluster
            n_samples: Number of samples to select
            
        Returns:
            List of indices of representative samples
        """
        # Dictionary mapping pattern types to key features and their expected direction
        pattern_features = {
            "Cloud Enhancement": {
                'kt': 'high',
                'kb': 'high',
                'kd': 'high',
                'clear_sky_score': 'high'
            },
            "Low Sun Angle": {
                'zenith': 'high',
                'direct_to_diffuse_ratio': 'low',
                'kt': 'low',
                'global_horizontal': 'low'
            },
            "Low Direct-to-Diffuse Ratio": {
                'direct_to_diffuse_ratio': 'low', 
                'kb': 'low',
                'kb_to_kd_ratio': 'low'
            }
        }
        
        # Extract the subset of X for the cluster samples
        # Convert numpy array indices to list for proper indexing
        cluster_indices_list = cluster_indices.tolist() if isinstance(cluster_indices, np.ndarray) else list(cluster_indices)
        
        # Check if indices are in the DataFrame
        valid_indices = [idx for idx in cluster_indices_list if idx in X.index]
        if not valid_indices:
            # If none of the indices are valid, return empty list or random indices
            print(f"Warning: No valid indices found for cluster with pattern '{pattern}'")
            return []
        
        # Get the subset of X for these indices
        X_subset = X.loc[valid_indices]
        
        # Get processed data for the subset
        X_processed = self._prepare_data(X_subset)
        
        # Select features based on pattern
        if pattern in pattern_features:
            feature_weights = {}
            for feature, direction in pattern_features[pattern].items():
                if feature in X_processed.columns:
                    # For each key feature, calculate how well each sample represents the pattern
                    values = X_processed[feature].values
                    if len(values) == 0 or (values.max() - values.min()) < 1e-10:
                        continue
                        
                    if direction == 'high':
                        # For 'high' features, higher values are more representative
                        normalized_values = (values - values.min()) / (values.max() - values.min())
                    else:
                        # For 'low' features, lower values are more representative
                        normalized_values = 1 - (values - values.min()) / (values.max() - values.min())
                    
                    feature_weights[feature] = normalized_values
            
            # Calculate overall representativeness score
            if feature_weights:
                # Combine scores from all features 
                features_df = pd.DataFrame(feature_weights, index=X_processed.index)
                representativeness = features_df.mean(axis=1)
                
                # Get top N representative samples
                sample_count = min(n_samples, len(representativeness))
                selected_indices = representativeness.nlargest(sample_count).index.tolist()
                
                return selected_indices
        
        # Fallback: if pattern not recognized or no suitable features, return random samples
        sample_count = min(n_samples, len(valid_indices))
        if sample_count == 0:
            return []
        random_indices = np.random.choice(valid_indices, size=sample_count, replace=False)
        return random_indices.tolist()

    def _perform_clustering(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], 
                        misclass_indices: np.ndarray, misclass_type: str, 
                        X_processed: pd.DataFrame, n_clusters: int,
                        feature_subset: Optional[List[str]], 
                        use_shap_values: bool, dim_method: str,
                        umap_n_neighbors: int, umap_min_dist: float,
                        random_state: int) -> Optional[Dict]:
        """
        Process a specific misclassification type with a given dimensionality reduction method
        
        Args:
            X: Original input features
            y: True labels
            misclass_indices: Indices of misclassifications in X
            misclass_type: Type of misclassification ('false_positives' or 'false_negatives')
            X_processed: Processed features
            n_clusters: Number of clusters to create
            feature_subset: Optional list of features to use for clustering
            use_shap_values: Whether to cluster based on SHAP values
            dim_method: Dimensionality reduction method ('pca' or 'umap')
            umap_n_neighbors: Number of neighbors for UMAP
            umap_min_dist: Minimum distance for UMAP
            random_state: Random seed for clustering
            
        Returns:
            Dictionary with clustering results or None if no misclassifications
        """
        print(f"\n===== CLUSTERING {misclass_type.upper()} with {dim_method.upper()} =====")
        
        # Get data for misclassifications
        X_misc = X_processed[misclass_indices].copy()
        
        # Get the datetime indices for these misclassifications
        datetime_indices = X.index[misclass_indices].tolist()

        print(f"Found {len(X_misc)} {misclass_type}")
        
        # Skip if no misclassifications
        if len(X_misc) == 0:
            return None
            
        # Prepare data for clustering
        if use_shap_values:
            # Compute SHAP values
            print(f"Computing SHAP values for {misclass_type}...")
            shap_values_misc = self.explainer(X_misc)
            
            # Use SHAP values for clustering
            cluster_data = pd.DataFrame(
                shap_values_misc.values,
                columns=self.feature_names,
                index=X_misc.index
            )
        else:
            # Use raw features
            cluster_data = X_misc
        
        # Filter to feature subset if specified
        if feature_subset:
            available_features = [f for f in feature_subset if f in cluster_data.columns]
            if not available_features:
                print("Warning: None of the specified features are available. Using all features.")
                available_features = cluster_data.columns.tolist()
            cluster_data = cluster_data[available_features]
        
        # Normalize data
        print(f"Normalizing data with {len(cluster_data.columns)} features...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Apply dimensionality reduction if needed for visualization or to reduce features
        if dim_method == 'pca':
            # PCA for dimensionality reduction
            print("Applying PCA for dimensionality reduction...")
            pca = PCA(n_components=min(scaled_data.shape[1], 10))  # Use up to 10 components
            reduced_data = pca.fit_transform(scaled_data)
            
            # Print explained variance
            explained_var = pca.explained_variance_ratio_.cumsum()
            print(f"Explained variance with {len(explained_var)} PCA components: {explained_var[-1]:.2f}")
            
        elif dim_method == 'umap':
            # UMAP for dimensionality reduction
            print(f"Applying UMAP (n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist})...")
            reducer = umap.UMAP(
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                n_components=min(scaled_data.shape[1], 10),  # Use up to 10 components
                random_state=random_state
            )
            reduced_data = reducer.fit_transform(scaled_data)
            
        else:
            # No dimensionality reduction
            reduced_data = scaled_data
        
        # Find optimal number of clusters if not specified
        if n_clusters is None:
            print("Finding optimal number of clusters...")
            silhouette_scores = []
            range_n_clusters = range(2, min(10, len(X_misc) // 10))
            for n in range_n_clusters:
                kmeans = KMeans(n_clusters=n, random_state=random_state, n_init=10)
                cluster_labels = kmeans.fit_predict(reduced_data)
                score = silhouette_score(reduced_data, cluster_labels)
                silhouette_scores.append(score)
                print(f"  {n} clusters: silhouette score = {score:.4f}")
            
            n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
            print(f"Optimal number of clusters: {n_clusters}")
        
        # Perform k-means clustering
        print(f"Performing k-means clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(reduced_data)
        
        # Add cluster labels to the data
        X_misc['cluster'] = cluster_labels
        
        # Calculate silhouette score
        sil_score = silhouette_score(reduced_data, cluster_labels)
        print(f"Silhouette score: {sil_score:.4f}")
        
        # Analyze clusters
        cluster_stats = self._analyze_clusters(X_misc, cluster_labels, n_clusters, X, y)
        
        # Visualize clusters - we need to adapt this to support UMAP visualization
        fig = self._visualize_clusters(
            X_misc, cluster_labels, n_clusters, cluster_stats, 
            f'{misclass_type.replace("_", " ").title()} with {dim_method.upper()}',
            reduced_data=reduced_data, 
            dim_method=dim_method
        )
        
        # Plot multiple forces for a few samples
        # if len(misclass_indices) > 0:
        #     sample_indices = misclass_indices[:5]
        #     print(f"\nPlotting SHAP force plots for the first {min(5, len(misclass_indices))} {misclass_type}...")
        #     self.plot_multiple_forces(X, sample_indices, y)
        
        # Return results
        return {
            'data': X_misc,
            'cluster_labels': cluster_labels,
            'cluster_stats': cluster_stats,
            'silhouette_score': sil_score,
            'figure': fig,
            'original_indices': misclass_indices,
            'datetime_indices': datetime_indices,
            'dim_reduction_method': dim_method,
            'reduced_data': reduced_data
        }


    def _analyze_clusters(self, X_clustered, cluster_labels, n_clusters, X_full, y_full):
        """
        Analyze the characteristics of each cluster.
        
        Args:
            X_clustered: Data with cluster labels
            cluster_labels: Cluster assignments
            n_clusters: Number of clusters
            X_full: Full dataset
            y_full: Full labels
            
        Returns:
            Dictionary with cluster statistics
        """
        
        print("X_clustered NaN check:")
        print(X_clustered.isna().sum()[X_clustered.isna().sum() > 0])
    

        # Important features to check based on SHAP summary
        key_features = [
            'direct_to_diffuse_ratio', 'kt', 'kb', 'kd', 
            'global_horizontal', 'direct_normal', 'diffuse_horizontal',
            'clear_sky_score', 'zenith', 'kb_to_kd_ratio',
            'clear_dni', 'clear_ghi', 'diffuse_enhancement'
        ]
        
        # Ensure we only analyze numeric features
        numeric_cols = X_clustered.select_dtypes(include=np.number).columns
        
        # Filter to available numeric features
        available_features = [f for f in key_features if f in numeric_cols]
        
        # If none of our preferred features are available, use whatever numeric features we have
        if not available_features:
            available_features = numeric_cols.tolist()
            if 'cluster' in available_features:
                available_features.remove('cluster')
        
        # Create a dictionary to store cluster statistics
        cluster_stats = {}
        
        # Calculate overall mean for comparison
        overall_mean = X_clustered[available_features].mean()
        
        # For each cluster, calculate statistics
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = X_clustered[cluster_mask]
            
            print(f"Cluster {cluster_id} shape: {cluster_data.shape}")
            print("Cluster data NaN check:")
            print(cluster_data.isna().sum()[cluster_data.isna().sum() > 0])
        
            # Basic statistics
            size = len(cluster_data)
            percent = size / len(X_clustered) * 100
            
            # Feature means
            means = cluster_data[available_features].mean()
            
            # Feature relative differences from overall mean
            # Avoid division by zero by replacing zeros with ones
            safe_overall_mean = overall_mean.replace(0, 1)
            rel_diff = (means - overall_mean) / safe_overall_mean * 100
            
            # Check for specific patterns
            has_cloud_enhancement = (means.get('kt', 0) > 1.0) if 'kt' in means else False
            has_low_direct_diffuse = (means.get('direct_to_diffuse_ratio', 1) < 0.2) if 'direct_to_diffuse_ratio' in means else False
            has_high_zenith = (means.get('zenith', 0) > 70) if 'zenith' in means else False
            
            # Store statistics
            cluster_stats[cluster_id] = {
                'size': size,
                'percent': percent,
                'means': means,
                'relative_diff': rel_diff,
                'has_cloud_enhancement': has_cloud_enhancement,
                'has_low_direct_diffuse': has_low_direct_diffuse,
                'has_high_zenith': has_high_zenith
            }
            
            # Try to categorize the cluster
            pattern_name = self._categorize_cluster_pattern(means, rel_diff)
            cluster_stats[cluster_id]['pattern'] = pattern_name
            
            # Print summary
            print(f"\nCluster {cluster_id}: {size} samples ({percent:.1f}%)")
            print(f"Pattern: {pattern_name}")
            print("Key feature means:")
            for feature in available_features[:10]:  # Show top 10 features
                print(f"  {feature}: {means[feature]:.4f} ({rel_diff[feature]:+.1f}% from mean)")
            
            # Print pattern indicators
            print("Pattern indicators:")
            if has_cloud_enhancement:
                print("  ✓ Shows cloud enhancement (kt > 1.0)")
            if has_low_direct_diffuse:
                print("  ✓ Has low direct-to-diffuse ratio (< 0.2)")
            if has_high_zenith:
                print("  ✓ High solar zenith angle (> 70°)")
            
            # Representative samples
            # Find the sample closest to the cluster centroid
            if size > 0 and len(available_features) > 0:
                valid_cluster_data = cluster_data.dropna()
                if len(valid_cluster_data) > 0:
                    # Calculate distances to cluster mean
                    dists = []
                    for idx, row in cluster_data[available_features].iterrows():
                        # print(f"\nChecking sample at index: {idx}")
                        # print("Sample NaN check:")
                        # print(row.isna().sum() if row.isna().sum() > 0 else "No NaNs in sample")
                        
                        dist = np.sqrt(((row - means) ** 2).sum())
                        dists.append((idx, dist))
                    
                    # Sort by distance
                    dists.sort(key=lambda x: x[1])
                    
                    # Get closest sample
                    if dists:
                        closest_idx = dists[0][0]
                        cluster_stats[cluster_id]['representative_idx'] = closest_idx
                        
                        # Print sample info using the clustered data instead of full data
                        print(f"\nRepresentative sample (index: {closest_idx}):")
                        # Use the processed data that we know contains the values
                        self._print_key_features(X_clustered.loc[closest_idx])
                        
                        # Add verification print to debug
                        print("\nVerification of available features:")
                        for feature in available_features:
                            value = X_clustered.loc[closest_idx, feature]
                            print(f"  {feature}: {value}")
                else:
                    print("\nWarning: No valid samples without NaN values found in cluster")
                
        return cluster_stats


    def _visualize_clusters(self, X_clustered, cluster_labels, n_clusters, cluster_stats, 
                        title_prefix='Clusters', reduced_data=None, dim_method='pca'):
        """
        Create visualizations of clusters with support for both PCA and UMAP.
        Support for displaying up to 6 cluster profiles (2 rows of 3).
        
        Args:
            X_clustered: Data with cluster labels
            cluster_labels: Cluster assignments
            n_clusters: Number of clusters
            cluster_stats: Statistics for each cluster
            title_prefix: Prefix for plot titles
            reduced_data: Pre-computed reduced data for visualization (for UMAP or PCA)
            dim_method: Dimensionality reduction method used ('pca' or 'umap')
            
        Returns:
            matplotlib.figure.Figure
        """
        
        # Select features for visualization
        vis_features = ['direct_to_diffuse_ratio', 'kt', 'kb', 'clear_sky_score', 'zenith']
        available_features = [f for f in vis_features if f in X_clustered.columns]
        
        if len(available_features) < 2:
            # Fall back to using the first two columns if our preferred features aren't available
            numeric_cols = X_clustered.select_dtypes(include=np.number).columns
            available_features = numeric_cols[:2] if len(numeric_cols) >= 2 else X_clustered.columns[:2]
        
        # Create figure with 4 rows instead of 3
        fig = plt.figure(figsize=(20, 20))  # Increased height to accommodate the extra row
        gs = gridspec.GridSpec(4, 3, figure=fig)  # Changed from 3x3 to 4x3
        
        # Plot 1: Scatter plot of two most important features
        ax1 = fig.add_subplot(gs[0, 0:2])
        
        feature_x = available_features[0] if len(available_features) > 0 else X_clustered.columns[0]
        feature_y = available_features[1] if len(available_features) > 1 else X_clustered.columns[1]
        
        scatter = ax1.scatter(
            X_clustered[feature_x], 
            X_clustered[feature_y],
            c=cluster_labels, 
            cmap='viridis', 
            alpha=0.7,
            s=50
        )
        
        # Add cluster centroids
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            centroid_x = X_clustered[cluster_mask][feature_x].mean()
            centroid_y = X_clustered[cluster_mask][feature_y].mean()
            
            # Add centroid marker
            ax1.scatter(centroid_x, centroid_y, s=200, c='red', marker='X', edgecolors='black', alpha=0.8)
            
            # Add cluster label with pattern
            pattern = cluster_stats[cluster_id]['pattern']
            ax1.annotate(
                f"Cluster {cluster_id}\n({pattern})",
                (centroid_x, centroid_y),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
        
        # Add reference lines if these are clearness features
        if feature_x == 'kt' or feature_y == 'kt':
            if feature_x == 'kt':
                ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, 
                        label='Cloud enhancement threshold (kt=1.0)')
            else:
                ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, 
                        label='Cloud enhancement threshold (kt=1.0)')
        
        if feature_x == 'direct_to_diffuse_ratio' or feature_y == 'direct_to_diffuse_ratio':
            ratio_threshold = 0.2
            if feature_x == 'direct_to_diffuse_ratio':
                ax1.axvline(x=ratio_threshold, color='blue', linestyle='--', alpha=0.7, 
                        label=f'Low direct-to-diffuse ratio threshold ({ratio_threshold})')
            else:
                ax1.axhline(y=ratio_threshold, color='blue', linestyle='--', alpha=0.7, 
                        label=f'Low direct-to-diffuse ratio threshold ({ratio_threshold})')
        
        # Add labels
        ax1.set_title(f"{title_prefix}: {feature_x.replace('_', ' ').title()} vs {feature_y.replace('_', ' ').title()}", fontsize=14)
        ax1.set_xlabel(feature_x.replace('_', ' ').title(), fontsize=12)
        ax1.set_ylabel(feature_y.replace('_', ' ').title(), fontsize=12)
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Cluster', fontsize=12)
        
        # Plot 2: PCA or UMAP visualization
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Try to get enough features for dimensionality reduction
        dim_features = X_clustered.select_dtypes(include=np.number).columns.tolist()
        # Remove 'cluster' column if present
        if 'cluster' in dim_features:
            dim_features.remove('cluster')
        
        if len(dim_features) >= 2:
            # Use pre-computed dimensionality reduction
            if reduced_data is not None and reduced_data.shape[1] >= 2:
                # Use the first two dimensions
                dim_result = reduced_data[:, :2]
                
                # Set appropriate titles based on dimensionality reduction method
                method_title = f"{dim_method.upper()} Visualization"
                xlabel = f"{dim_method.upper()} Component 1"
                ylabel = f"{dim_method.upper()} Component 2"
                
                # Plot dimensionality reduction results
                scatter_dim = ax2.scatter(
                    dim_result[:, 0], 
                    dim_result[:, 1],
                    c=cluster_labels, 
                    cmap='viridis', 
                    alpha=0.7,
                    s=50
                )
                
                # Add cluster centroids in reduced space
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    centroid_x = dim_result[cluster_mask, 0].mean()
                    centroid_y = dim_result[cluster_mask, 1].mean()
                    
                    # Add centroid marker
                    ax2.scatter(centroid_x, centroid_y, s=200, c='red', marker='X', edgecolors='black', alpha=0.8)
                    
                    # Add cluster label
                    ax2.annotate(
                        f"Cluster {cluster_id}",
                        (centroid_x, centroid_y),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    )
                
                # Add labels
                ax2.set_title(method_title, fontsize=14)
                ax2.set_xlabel(xlabel, fontsize=12)
                ax2.set_ylabel(ylabel, fontsize=12)
                ax2.grid(alpha=0.3)
            else:
                ax2.text(0.5, 0.5, f"No dimensionality reduction data available", 
                        ha='center', va='center', fontsize=14)
        else:
            ax2.text(0.5, 0.5, f"Not enough numeric features for {dim_method.upper()}", 
                    ha='center', va='center', fontsize=14)
        
        # Feature subset for parallel coordinates plots
        feature_subset = ['direct_to_diffuse_ratio', 'kt', 'kb', 'kd', 'clear_sky_score', 
                    'zenith', 'global_horizontal', 'direct_normal', 'diffuse_horizontal']
        # Make sure we only use numeric features
        numeric_cols = X_clustered.select_dtypes(include=np.number).columns
        available_subset = [f for f in feature_subset if f in numeric_cols]
        
        # If we don't have enough of our preferred features, use whatever numeric features are available
        if len(available_subset) < 3:
            available_subset = numeric_cols[:8].tolist()
            if 'cluster' in available_subset:
                available_subset.remove('cluster')
        
        # Plot 3-5: First row of cluster profiles (clusters 0-2)
        for i, cluster_id in enumerate(range(min(3, n_clusters))):
            ax = fig.add_subplot(gs[1, i])
            
            # Get cluster data
            cluster_mask = cluster_labels == cluster_id
            
            if cluster_mask.sum() > 0 and len(available_subset) > 0:
                cluster_data = X_clustered[cluster_mask]
                
                # Normalize data for parallel coordinates
                normalized_data = cluster_data[available_subset].copy()
                for feature in available_subset:
                    min_val = normalized_data[feature].min()
                    max_val = normalized_data[feature].max()
                    if max_val > min_val:
                        normalized_data[feature] = (normalized_data[feature] - min_val) / (max_val - min_val)
                
                # Plot parallel coordinates
                pd.plotting.parallel_coordinates(
                    normalized_data.assign(cluster=cluster_id), 
                    'cluster',
                    ax=ax,
                    color=['blue'],
                    alpha=0.05
                )
                
                # Add mean profile with thicker line
                mean_profile = normalized_data.mean()
                x = list(range(len(available_subset)))
                y = [mean_profile[feat] for feat in available_subset]
                ax.plot(x, y, color='red', linewidth=3, label='Mean')
                
                # Customize plot
                ax.set_title(f"Cluster {cluster_id} Profile ({cluster_stats[cluster_id]['pattern']})", fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels([f.replace('_', ' ').title() for f in available_subset], rotation=45, ha='right')
                ax.set_ylim(-0.05, 1.05)
                ax.legend()
            else:
                ax.text(0.5, 0.5, f"No data for Cluster {cluster_id}" if cluster_mask.sum() > 0 else "No numeric features available", 
                    ha='center', va='center', fontsize=14)
        
        # Plot 6-8: Second row of cluster profiles (clusters 3-5)
        for i, cluster_id in enumerate(range(3, min(6, n_clusters))):
            ax = fig.add_subplot(gs[2, i])
            
            # Get cluster data
            cluster_mask = cluster_labels == cluster_id
            
            if cluster_mask.sum() > 0 and len(available_subset) > 0:
                cluster_data = X_clustered[cluster_mask]
                
                # Normalize data for parallel coordinates
                normalized_data = cluster_data[available_subset].copy()
                for feature in available_subset:
                    min_val = normalized_data[feature].min()
                    max_val = normalized_data[feature].max()
                    if max_val > min_val:
                        normalized_data[feature] = (normalized_data[feature] - min_val) / (max_val - min_val)
                
                # Plot parallel coordinates
                pd.plotting.parallel_coordinates(
                    normalized_data.assign(cluster=cluster_id), 
                    'cluster',
                    ax=ax,
                    color=['blue'],
                    alpha=0.05
                )
                
                # Add mean profile with thicker line
                mean_profile = normalized_data.mean()
                x = list(range(len(available_subset)))
                y = [mean_profile[feat] for feat in available_subset]
                ax.plot(x, y, color='red', linewidth=3, label='Mean')
                
                # Customize plot
                ax.set_title(f"Cluster {cluster_id} Profile ({cluster_stats[cluster_id]['pattern']})", fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels([f.replace('_', ' ').title() for f in available_subset], rotation=45, ha='right')
                ax.set_ylim(-0.05, 1.05)
                ax.legend()
            else:
                ax.text(0.5, 0.5, f"No data for Cluster {cluster_id}" if cluster_mask.sum() > 0 else "No numeric features available", 
                    ha='center', va='center', fontsize=14)
        
        # Plot 9: Cluster size comparison (moved to fourth row)
        ax9 = fig.add_subplot(gs[3, 0])
        
        # Create bar chart of cluster sizes
        sizes = [cluster_stats[i]['size'] for i in range(n_clusters)]
        patterns = [cluster_stats[i]['pattern'] for i in range(n_clusters)]
        
        bars = ax9.bar(range(n_clusters), sizes)
        
        # Add percentage labels
        total = sum(sizes)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax9.annotate(
                f'{height/total*100:.1f}%',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom'
            )
        
        # Add pattern labels
        ax9.set_xticks(range(n_clusters))
        labels = [f"{i}\n({p})" for i, p in enumerate(patterns)]
        ax9.set_xticklabels(labels)
        plt.setp(ax9.get_xticklabels(), rotation=45, ha='right')

        # Add title and labels
        ax9.set_title(f"{title_prefix}: Cluster Sizes", fontsize=14)
        ax9.set_xlabel("Cluster", fontsize=12)
        ax9.set_ylabel("Number of Samples", fontsize=12)
        ax9.grid(axis='y', alpha=0.3)
        
        # Plot 10: Feature importance by cluster (heatmap) (moved to fourth row)
        ax10 = fig.add_subplot(gs[3, 1:])
        
        # Make sure to only use numeric features for the heatmap
        numeric_subset = [f for f in available_subset if f in numeric_cols]
        
        if numeric_subset and n_clusters > 0:
            # Create data for heatmap
            heatmap_data = []
            for cluster_id in range(n_clusters):
                relative_diffs = cluster_stats[cluster_id]['relative_diff']
                # Make sure we only include features that are in relative_diffs
                available_diffs = [relative_diffs.get(f, 0) for f in numeric_subset if f in relative_diffs]
                # If we don't have enough features, pad with zeros
                if len(available_diffs) < len(numeric_subset):
                    heatmap_data.append(available_diffs + [0] * (len(numeric_subset) - len(available_diffs)))
                else:
                    heatmap_data.append(available_diffs)
            
            # Create heatmap only if we have data
            if heatmap_data and any(row for row in heatmap_data):
                heatmap = sns.heatmap(
                    heatmap_data, 
                    annot=True, 
                    fmt=".1f", 
                    cmap="coolwarm", 
                    center=0,
                    ax=ax10,
                    yticklabels=[f"{i} ({patterns[i]})" for i in range(n_clusters)],
                    xticklabels=[f.replace('_', ' ').title() for f in numeric_subset]
                )
                
                # Customize heatmap
                ax10.set_title(f"{title_prefix}: Feature Deviation from Mean by Cluster (%)", fontsize=14)
                ax10.set_xlabel("Feature", fontsize=12)
                ax10.set_ylabel("Cluster", fontsize=12)
                plt.setp(ax10.get_xticklabels(), rotation=45, ha='right')
            else:
                ax10.text(0.5, 0.5, "Insufficient data for heatmap", 
                        ha='center', va='center', fontsize=14)
        else:
            ax10.text(0.5, 0.5, "No numeric features available for heatmap", 
                    ha='center', va='center', fontsize=14)
        
        # Adjust layout and add overall title
        plt.suptitle(f"{title_prefix}", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Display the figure explicitly
        plt.show()
        
        return fig


    def _categorize_cluster_pattern(self, means, rel_diff):
        """
        Categorize a cluster based on its mean feature values.
        
        Args:
            means: Mean values for each feature
            rel_diff: Relative difference from overall mean
            
        Returns:
            String describing the pattern
        """
        # Check for cloud enhancement
        if 'kt' in means and means['kt'] > 1.0:
            return "Cloud Enhancement"
        
        # Check for low direct-to-diffuse ratio
        if 'direct_to_diffuse_ratio' in means and means['direct_to_diffuse_ratio'] < 0.2:
            return "Low Direct-to-Diffuse Ratio"
        
        # Check for high zenith angle (low sun)
        if 'zenith' in means and means['zenith'] > 70:
            return "Low Sun Angle"
        
        # Check for high diffuse enhancement
        if 'diffuse_enhancement' in means and means['diffuse_enhancement'] > 1.2:
            return "Diffuse Enhancement"
        
        # Check for high direct normal but low global
        if ('direct_normal' in means and 'global_horizontal' in means and
            rel_diff['direct_normal'] > 20 and rel_diff['global_horizontal'] < -20):
            return "High Direct, Low Global"
        
        # Check for inconsistent clearness indices
        if ('kt' in means and 'kb' in means and 'kd' in means and
            abs(rel_diff['kt'] - rel_diff['kb']) > 30):
            return "Inconsistent Clearness Indices"
        
        # Default pattern if no specific pattern is detected
        return "Mixed Pattern"
    
 

    def plot_false_cases_on_timeseries(self, X: pd.DataFrame, y: pd.Series, 
                                    cluster_results: Dict,
                                    init_start_date: str, window_size: int = 30,
                                    misclassification_type: str = 'false_positives',
                                    dim_method: str = 'umap',
                                    include_vars: List[str] = None,
                                    dump_html: bool = True,
                                    dump_dir: str = 'outputs/'):
        """
        Plot misclassification cases on the original time series data, color-coded by cluster.
        
        Args:
            X: Original test dataset with DatetimeIndex
            y: True labels
            cluster_results: Results from cluster_misclassifications
            init_start_date: Initial start date for the plot window (e.g. '2015-01-01')
                            If None, will use the first date in X
            window_size: Number of days to include in each plot window
            misclassification_type: Type of misclassifications to plot ('false_positives', 'false_negatives', or 'both')
            dim_method: Dimensionality reduction method used for clustering ('umap' or 'pca')
            include_vars: List of variables to include in the plot
                        (default: direct_normal, global_horizontal, kt, kb)
            dump_html: Whether to save the plots as HTML files
            dump_dir: Directory to save HTML files
        
        Returns:
            Dictionary of plotly figure objects by date range
        """
        
        # Default variables to plot if not specified
        if include_vars is None:
            include_vars = ['direct_normal', 'global_horizontal', 'kt', 'kb']
        
        # Create output directory if it doesn't exist
        if dump_html and not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        
        # Check if X.index is timezone-aware
        is_tz_aware = X.index.tzinfo is not None
        
        # Debug statements to understand the data
        print(f"X.index is timezone-aware: {is_tz_aware}")
        if is_tz_aware:
            print(f"X.index timezone: {X.index[0].tzinfo}")
        
        # Use the full dataset but create multiple plots with window_size days each
        if init_start_date is None:
            start_date = X.index.min().normalize()  # Start from beginning of data
        else:
            # Parse the initial start date and handle timezone
            start_date = pd.to_datetime(init_start_date)
            # If X has timezone but start_date doesn't, localize start_date to X's timezone
            if is_tz_aware and start_date.tzinfo is None:
                # Get timezone from X
                tz = X.index[0].tzinfo
                print(f"Localizing start_date to timezone: {tz}")
                start_date = start_date.tz_localize(tz)
            start_date = start_date.normalize()  # Normalize to midnight
        
        print(f"Parsed start_date: {start_date}")
        print(f"X.index.min(): {X.index.min()}")
        
        # Ensure start_date is within dataset range
        if start_date < X.index.min():
            start_date = X.index.min().normalize()
            print(f"Adjusted start_date to X.index.min(): {start_date}")
        
        end_date = X.index.max().normalize()
        print(f"End date: {end_date}")
        
        # Get model predictions for the entire dataset at once
        print("Computing model predictions...")
        # X = self.model._prepare_features(X, is_training=False)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Identify all misclassifications
        false_positives = (y_pred == 1) & (y == 0)
        false_negatives = (y_pred == 0) & (y == 1)
        
        # Determine which misclassifications to plot
        if misclassification_type == 'false_positives':
            misclass_mask = false_positives
            title_prefix = "False Positives"
        elif misclassification_type == 'false_negatives':
            misclass_mask = false_negatives
            title_prefix = "False Negatives"
        elif misclassification_type == 'both':
            misclass_mask = false_positives | false_negatives
            title_prefix = "Misclassifications"
        else:
            raise ValueError("misclassification_type must be 'false_positives', 'false_negatives', or 'both'")
        
        # Get all misclassification indices
        all_misclass_indices = X[misclass_mask].index
        print(f"Found {len(all_misclass_indices)} total {misclassification_type} in the dataset")
        
        # Extract clustering results for the specified method and misclassification type
        cluster_key_mapping = {
            'false_positives': f'false_positives_{dim_method}',
            'false_negatives': f'false_negatives_{dim_method}',
            'both': None  # Special handling for 'both'
        }
        
        # Create mapping of timestamps to cluster assignments
        timestamp_to_cluster = {}
        
        if misclassification_type != 'both':
            # Handle single misclassification type
            cluster_key = cluster_key_mapping[misclassification_type]
            
            # Try different ways to access the cluster results
            if cluster_key in cluster_results:
                results = cluster_results[cluster_key]
            elif misclassification_type in cluster_results and cluster_results[misclassification_type].get('dim_reduction_method') == dim_method:
                results = cluster_results[misclassification_type]
            else:
                print(f"Warning: No {dim_method} clustering results found for {misclassification_type}")
                results = None
            
            if results:
                cluster_data = results['data']
                cluster_labels = results['cluster_labels']
                cluster_stats = results['cluster_stats']
                n_clusters = len(cluster_stats)
                
                print(f"Found {n_clusters} clusters for {misclassification_type}")
                
                # Define cluster colors - ensure each cluster gets a unique color
                cluster_colors = [
                    'red', 'blue', 'green', 'purple', 'orange', 
                    'cyan', 'magenta', 'lime', 'brown', 'pink'
                ]
                
                # Map timestamps to clusters
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_samples = cluster_data[cluster_mask]
                    pattern = cluster_stats[cluster_id]['pattern']
                    
                    print(f"Cluster {cluster_id} ({pattern}) has {len(cluster_samples)} samples")
                    
                    # Get timestamps for this cluster
                    for idx in cluster_samples.index:
                        if idx in all_misclass_indices:  # Only include if it's a misclassification
                            timestamp_to_cluster[idx] = {
                                'cluster': cluster_id,
                                'pattern': pattern,
                                'type': misclassification_type,
                                'color': cluster_colors[cluster_id % len(cluster_colors)]
                            }
        else:
            # Handle both misclassification types
            for mc_type in ['false_positives', 'false_negatives']:
                cluster_key = f'{mc_type}_{dim_method}'
                
                # Try different ways to access the cluster results
                if cluster_key in cluster_results:
                    results = cluster_results[cluster_key]
                elif mc_type in cluster_results and cluster_results[mc_type].get('dim_reduction_method') == dim_method:
                    results = cluster_results[mc_type]
                else:
                    print(f"Warning: No {dim_method} clustering results found for {mc_type}")
                    continue
                
                cluster_data = results['data']
                cluster_labels = results['cluster_labels']
                cluster_stats = results['cluster_stats']
                n_clusters = len(cluster_stats)
                
                print(f"Found {n_clusters} clusters for {mc_type}")
                
                # Define cluster colors - different sets for FP and FN
                fp_colors = ['red', 'blue', 'green', 'purple', 'orange']
                fn_colors = ['darkred', 'darkblue', 'darkgreen', 'darkviolet', 'darkorange']
                
                # Use appropriate color set
                cluster_colors = fp_colors if mc_type == 'false_positives' else fn_colors
                
                # Map timestamps to clusters
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    cluster_samples = cluster_data[cluster_mask]
                    pattern = cluster_stats[cluster_id]['pattern']
                    
                    print(f"{mc_type} Cluster {cluster_id} ({pattern}) has {len(cluster_samples)} samples")
                    
                    # Get timestamps for this cluster
                    for idx in cluster_samples.index:
                        if idx in all_misclass_indices:  # Only include if it's a misclassification
                            timestamp_to_cluster[idx] = {
                                'cluster': cluster_id,
                                'pattern': pattern,
                                'type': mc_type,
                                'color': cluster_colors[cluster_id % len(cluster_colors)]
                            }
        
        print(f"Found {len(timestamp_to_cluster)} timestamps with cluster assignments")
        
        # Create multiple plots for each time window
        figures = {}
        current_start = start_date
        
        while current_start <= end_date:
            current_end = current_start + pd.Timedelta(days=window_size)
            
            # Format dates as strings for filtering
            start_str = current_start.strftime('%Y-%m-%d')
            end_str = current_end.strftime('%Y-%m-%d')
            
            print(f"Processing window {start_str} to {end_str}")
            
            # Filter the time series to the desired date range - handle potential timezone issues
            X_window = X.loc[current_start:current_end].copy()
            
            if len(X_window) == 0:
                print(f"No data found between {start_str} and {end_str}")
                current_start = current_end
                continue
                
            # Get corresponding labels
            y_window = y.loc[X_window.index].copy()
            
            # Get misclassifications in this window
            window_indices = X_window.index
            window_fp = false_positives.loc[window_indices]
            window_fn = false_negatives.loc[window_indices]
            
            if misclassification_type == 'false_positives':
                window_misclass = window_indices[window_fp]
            elif misclassification_type == 'false_negatives':
                window_misclass = window_indices[window_fn]
            else:  # 'both'
                window_misclass = window_indices[window_fp | window_fn]
            
            print(f"Window has {len(window_misclass)} {misclassification_type}")
            
            if len(window_misclass) == 0:
                print(f"No {misclassification_type} found in this window, skipping")
                current_start = current_end
                continue
            
            # Create a figure for this window
            fig = go.Figure()
            
            # Plot each irradiance variable as a line
            for var in include_vars:
                if var in X_window.columns:
                    fig.add_trace(go.Scatter(
                        x=X_window.index,
                        y=X_window[var],
                        mode='lines',
                        name=var,
                        line=dict(width=1)
                    ))
            
            # Add labeled clear sky periods as yellow highlighted regions
            clearsky_periods = []
            if 'clearsky_label' in X_window.columns:
                clearsky = X_window['clearsky_label'] == 1
            else:
                clearsky = y_window == 1
            
            # Find continuous periods of clearsky
            current_period = None
            for i, (idx, is_clear) in enumerate(clearsky.items()):
                if is_clear and current_period is None:
                    current_period = {'start': idx}
                elif not is_clear and current_period is not None:
                    current_period['end'] = X_window.index[i-1]
                    clearsky_periods.append(current_period)
                    current_period = None
            
            # Add the last period if it extends to the end
            if current_period is not None:
                current_period['end'] = X_window.index[-1]
                clearsky_periods.append(current_period)
            
            # Add highlighted regions for clear sky periods
            for period in clearsky_periods:
                fig.add_vrect(
                    x0=period['start'],
                    x1=period['end'],
                    fillcolor="yellow",
                    opacity=0.2,
                    layer="below",
                    line_width=0
                )
            
            # Group points by cluster
            cluster_points = {}
            for idx in window_misclass:
                if idx in timestamp_to_cluster:
                    info = timestamp_to_cluster[idx]
                    cluster_id = info['cluster']
                    mc_type = info['type']
                    
                    key = (mc_type, cluster_id)
                    if key not in cluster_points:
                        cluster_points[key] = {
                            'indices': [],
                            'pattern': info['pattern'],
                            'color': info['color'],
                            'type': mc_type
                        }
                    
                    cluster_points[key]['indices'].append(idx)
            
            # Add traces for each cluster
            for (mc_type, cluster_id), data in cluster_points.items():
                pattern = data['pattern']
                points = data['indices']
                color = data['color']
                
                if not points:
                    continue
                
                # Set symbol based on misclassification type
                symbol = 'circle' if mc_type == 'false_positives' else 'diamond'
                
                # Determine display name
                prefix = "FP" if mc_type == 'false_positives' else "FN"
                name = f"{prefix} Cluster {cluster_id}: {pattern}"
                
                # Create hover text with details
                hover_texts = []
                for idx in points:
                    text = f"{name}<br>"
                    text += f"Time: {idx.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                    for var in include_vars:
                        if var in X_window.columns:
                            text += f"{var}: {X_window.loc[idx, var]:.2f}<br>"
                    hover_texts.append(text)
                
                # Add scatter points for this cluster
                fig.add_trace(go.Scatter(
                    x=points,
                    # Use first variable for y-values
                    y=[X_window.loc[idx, include_vars[0]] for idx in points],
                    mode='markers',
                    name=name,
                    marker=dict(
                        size=3,
                        color=color,
                        symbol=symbol,
                        line=dict(width=0, color=color)
                    ),
                    text=hover_texts,
                    hoverinfo='text'
                ))
            
            # Update layout
            fig.update_layout(
                title=f"1-Minute Irradiance Data over Time: Labeled Clearsky Periods in Yellow.<br>{title_prefix} Marked by {dim_method.upper()} Cluster",
                title_font=dict(size=10),
                xaxis_title="Time",
                yaxis_title="Irradiance (W/m²)",
                # legend=dict(
                #     orientation="h",
                #     yanchor="bottom",
                #     y=1.02,
                #     xanchor="right",
                #     x=1
                # ),
                showlegend=True,
                width=1200,
                height=600,
                margin=dict(l=50, r=50, t=100, b=50)
            )
            
            # Add range slider
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
            
            # Save HTML file if requested
            if dump_html:
                date_str = f"{current_start.strftime('%Y%m%d')}-{current_end.strftime('%Y%m%d')}"
                filename = f"{misclassification_type}_{dim_method}_{date_str}.html"
                filepath = os.path.join(dump_dir, filename)
                
                fig.write_html(filepath)
                print(f"Saved plot to {filepath}")
            
            # Store the figure
            figures[(start_str, end_str)] = fig
            
            # Move to next window
            current_start = current_end
        
        # Return the dictionary of figures or the last figure
        if len(figures) == 0:
            print("No figures were created")
            return None
            
        # For convenience, return the last figure for display
        return list(figures.values())[-1]


    def analyze_temporal_sequences(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], 
                                misclassification_results: Dict,
                                window_size: int = 30) -> Dict:
        """
        Compare temporal sequences between false positives and true positives.
        
        Args:
            X: Input features DataFrame
            y: True labels
            misclassification_results: Results from analyze_misclassifications
            window_size: Number of minutes before/after to analyze
            
        Returns:
            Dictionary containing analysis results and visualizations
        """
        # Get true positive indices
        X_processed, _ = self.model._prepare_features(X, is_training=False)
        y_pred = self.model.predict(X)
        true_positive_mask = (y_pred == 1) & (y == 1)
        true_positive_indices = np.where(true_positive_mask)[0]
        
        # Key features to track
        key_features = {
            'irradiance': [
                'global_horizontal', 'direct_normal', 'diffuse_horizontal',
                'clear_ghi', 'clear_dni', 'clear_dhi'
            ],
            'ratios': [
                'kt', 'kb', 'kd',
                'direct_to_diffuse_ratio',
                'diffuse_to_global_ratio',
                'direct_diffuse_ratio_match'
            ],
            'stability': [
                'kt_stability_5',
                'kb_stability_5',
                'kd_stability_5'
            ],
            'consistency': [
                'clear_sky_score',
            ]
        }

        results = {}
        
        # Process both false positives and true positives
        case_indices = {
            'false_positives': misclassification_results['false_positives']['indices'],
            'true_positives': true_positive_indices
        }
        plt.close('all')
        for category, features in key_features.items():
            print(f"Analyzing {category} features")
            fig = plt.figure(figsize=(15, 6*len(features)))
            fig.suptitle(f'Feature Patterns Comparison - {category}')
            
            category_stats = {}
            
            for i, feature in enumerate(features, 1):
                if feature not in X_processed.columns:
                    print(f"DEBUG: Feature '{feature}' not in X_processed.columns: {X_processed.columns}")
                    continue
                    
                ax = fig.add_subplot(len(features), 1, i)
                print(f"DEBUG: Created subplot for '{feature}'")

                feature_stats = {}
                
                for case_type, indices in case_indices.items():
                    sequences = []
                    all_values = []
                    all_times = []
                    
                    print(f"DEBUG: Processing {len(indices)} {case_type} indices for feature '{feature}'")
                    
                    for idx in indices:
                        # Extract window around the point
                        start_idx = max(0, idx - window_size)
                        end_idx = min(len(X_processed), idx + window_size)
                        sequence = X_processed.iloc[start_idx:end_idx].copy()
                        
                        # Check if feature has valid values
                        if feature not in sequence.columns:
                            print(f"DEBUG: Feature '{feature}' unexpectedly missing from sequence")
                            continue
                            
                        if sequence[feature].isna().all():
                            print(f"DEBUG: All values for feature '{feature}' are NaN in sequence")
                            continue

                        # Center the time axis
                        times = range(-min(window_size, idx), 
                                    min(window_size, len(X_processed) - idx))
                        values = sequence[feature].values
                        
                        all_values.extend(values)
                        all_times.extend(times)
                        sequences.append(pd.DataFrame({
                            'relative_time': times,
                            'value': values
                        }))
                    
                    if not sequences:
                        print(f"DEBUG: No valid sequences for {case_type} for feature '{feature}'")
                        continue
                    
                    # Combine all sequences
                    combined_seq = pd.concat(sequences)
                    
                    # Calculate statistics
                    point_stats = combined_seq[combined_seq['relative_time'] == 0]['value'].describe()
                    pre_point = combined_seq[(combined_seq['relative_time'] >= -5) & 
                                        (combined_seq['relative_time'] < 0)]
                    
                    if len(pre_point) > 1:
                        mean_values = pre_point.groupby('relative_time')['value'].mean()
                        rate_of_change = (mean_values.iloc[-1] - mean_values.iloc[0]) / 5
                        variability = pre_point['value'].std() / (pre_point['value'].mean() + 1e-10)
                    else:
                        rate_of_change = None
                        variability = None
                    
                    feature_stats[case_type] = {
                        'point_stats': point_stats,
                        'rate_of_change': rate_of_change,
                        'variability': variability
                    }
                    
                    # Plot sequences
                    color = 'red' if case_type == 'false_positives' else 'blue'
                    alpha = 0.05 if len(indices) > 100 else 0.1
                    
                    # Plot individual sequences
                    for seq in sequences:
                        ax.plot(seq['relative_time'], seq['value'], 
                            alpha=alpha, color=color)
                    
                    # Calculate and plot mean and std
                    mean_seq = combined_seq.groupby('relative_time')['value'].mean()
                    std_seq = combined_seq.groupby('relative_time')['value'].std()
                    
                    ax.plot(mean_seq.index, mean_seq.values, 
                        color=color, linewidth=2, 
                        label=f'{case_type.replace("_", " ").title()} Mean')
                    ax.fill_between(mean_seq.index, 
                                mean_seq.values - std_seq.values,
                                mean_seq.values + std_seq.values,
                                color=color, alpha=0.2)
                
                ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
                ax.set_xlabel('Minutes Relative to Classification')
                ax.set_ylabel(feature)
                ax.set_title(f'{feature} Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add statistical comparison text
                stats_text = []
                for case_type, stats in feature_stats.items():
                    stats_text.append(f"{case_type.replace('_', ' ').title()}:")
                    stats_text.append(f"Mean at point: {stats['point_stats']['mean']:.2f}")
                    stats_text.append(f"Std at point: {stats['point_stats']['std']:.2f}")
                    if stats['rate_of_change'] is not None:
                        stats_text.append(f"Rate of change: {stats['rate_of_change']:.4f}")
                    if stats['variability'] is not None:
                        stats_text.append(f"Variability: {stats['variability']:.4f}")
                    stats_text.append("")
                
                ax.text(1.02, 0.98, '\n'.join(stats_text),
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                category_stats[feature] = feature_stats
            
            plt.tight_layout()
            plt.savefig(f'outputs/feature_patterns_comparison_{category}.png')
            plt.show()

            results[category] = {
                # 'figure': fig,
                'statistics': category_stats
            }
        
        return results




