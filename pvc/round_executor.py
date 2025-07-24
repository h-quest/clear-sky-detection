import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
import numpy as np

def load_raw_data(data_path: str, timezone: str) -> pd.DataFrame:
    # Load data
    df = pd.read_csv(data_path)
    df['measured_on'] = pd.to_datetime(df['measured_on'])
    ts_col = df['measured_on']

    aware_times = []
    for ts in ts_col:
        # Directly localize to the target timezone
        try:
            aware_ts = pd.Timestamp(ts).tz_localize(timezone,
                                                    ambiguous=True,
                                                    nonexistent='shift_forward')
            aware_times.append(aware_ts)
        except Exception as e:
            # Handle potential errors (e.g., DST transitions)
            print(f"Warning: Error localizing timestamp {ts}: {e}")
            # Fall back to UTC localization if direct localization fails
            aware_ts = pd.Timestamp(ts).tz_localize('UTC').tz_convert(timezone)
            aware_times.append(aware_ts)

    df['datetime'] = pd.DatetimeIndex(aware_times)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    return df

def analyze_confusion_matrix(model, X_test, y_test):
    """Detailed analysis of confusion matrix with rates."""
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Extract values
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Clear', 'Clear'], 
                yticklabels=['Non-Clear', 'Clear'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Print detailed metrics
    print(f"Total samples: {total}")
    print(f"True Negatives: {tn} ({tn/total:.2%})")
    print(f"False Positives: {fp} ({fp/total:.2%})")
    print(f"False Negatives: {fn} ({fn/total:.2%})")
    print(f"True Positives: {tp} ({tp/total:.2%})")
    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")
    print(f"False Negative Rate: {false_negative_rate:.4f}")
    
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Compute classification metrics for clear sky detection.
    
    Args:
        y_true: Array of true labels (0 = not clear sky, 1 = clear sky)
        y_pred: Array of predicted labels (0 = not clear sky, 1 = clear sky)
        y_proba: Optional array of probability predictions for the positive class
        
    Returns:
        Dictionary of metrics
    """
    
    # Calculate basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Calculate confusion matrix and derived metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    
    metrics.update({
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'false_positive_rate': fp / (tn + fp) if (tn + fp) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'true_negative_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'tn_percent': 100 * tn / total if total > 0 else 0,
        'fp_percent': 100 * fp / total if total > 0 else 0,
        'fn_percent': 100 * fn / total if total > 0 else 0,
        'tp_percent': 100 * tp / total if total > 0 else 0,
    })
    
    # AUC metrics if probabilities provided
    if y_proba is not None:
        # ROC AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = float('nan')
            
        # Precision-Recall AUC
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            metrics['pr_auc'] = auc(recall, precision)
        except:
            metrics['pr_auc'] = float('nan')
    
    return metrics

def train_clear_sky_model(data_path: str, timezone: str, model):
    """
    Train a clear sky classifier on the given dataset.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Trained ClearSkyClassifier
    """

    # Load data
    df = load_raw_data(data_path, timezone)

    # Set up time-based train-test split
    # Use 80% for training, 20% for testing
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Prepare features and target
    X_train = train_df.drop('clearsky_label', axis=1)
    y_train = train_df['clearsky_label']
    
    X_test = test_df.drop('clearsky_label', axis=1)
    y_test = test_df['clearsky_label']
    
    # Initialize and train the model
    # model = ClearSkyClassifier(
    #     model_params={
    #         'iterations': 1000,
    #         'learning_rate': 0.05,  
    #         'depth': 5, # 6
    #         'l2_leaf_reg': 3,
    #         'eval_metric': 'F1',
    #         'early_stopping_rounds': 150,
    #         'loss_function': 'Logloss',
    #         'auto_class_weights': 'Balanced',
    #         'random_seed': 41,
    #         'verbose': 100
    #     },
        
    #     feature_engineering=True,
    #     handle_imbalance=True,
    #     latitude=39.7420,   
    #     longitude=-105.1778,
    #     altitude=1829,
    #     timezone='America/Denver'
    # )
    
    # Create a validation set for early stopping
    # Use the last 20% of training data as validation
    val_split_idx = int(len(X_train) * 0.8)
    X_val = X_train.iloc[val_split_idx:]
    y_val = y_train.iloc[val_split_idx:]
    
    # Fit the model
    model.fit(
        X_train, 
        y_train,
        eval_set=(X_val, y_val)
    )
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    model.plot_feature_importance(top_n=50)
    # model.plot_confusion_matrix(X_test, y_test)
    model.plot_precision_recall_curve(X_test, y_test)

    print("\nAnalyzing model performance and errors...")
    analyze_confusion_matrix(model, X_test, y_test)


    return model, X_test, y_test



# data_path = "./Clearsky-Labeled-Data-Set/labeled_clearsky_data.csv"
# model, X_test, y_test = train_clear_sky_model(data_path)

# Save the model
# model.save_model("clear_sky_classifier.cbm")
