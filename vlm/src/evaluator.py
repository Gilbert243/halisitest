from tqdm import tqdm
import time
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
from .utils import encode_image

class Evaluator:
    """Evaluator for hair classification models.
    
    Performs comprehensive evaluation including:
    - Accuracy metrics (overall, top-k)
    - Per-class metrics (precision, recall, F1)
    - Confusion matrix
    - Latency analysis (mean, median, percentiles)
    - Error analysis
    """
    
    def __init__(self, classifier):
        """Initialize evaluator with a classifier.
        
        Args:
            classifier: HairClassifier instance to evaluate
        """
        self.classifier = classifier

    def run(self, dataset_path, max_samples=None, verbose=True, save_results=True):
        """Run comprehensive evaluation on dataset.
        
        Args:
            dataset_path: Path to CSV file with image_path and type columns
            max_samples: Optional limit on number of samples to evaluate
            verbose: Show progress bar with stats
            save_results: Save detailed results to CSV
            
        Returns:
            Dictionary containing:
            - accuracy: Overall accuracy
            - weighted_f1: Weighted F1 score
            - macro_f1: Macro-averaged F1 score
            - confusion_matrix: Confusion matrix as array
            - classification_report: Per-class metrics
            - latency_stats: Latency statistics (mean, median, p95, p99)
            - latency_times: List of all latency measurements
            - results: DataFrame with predictions and ground truth
            - error_analysis: DataFrame of misclassified samples
        """
        # Load dataset
        df = pd.read_csv(dataset_path)
        if max_samples:
            df = df.head(max_samples)

        preds, times, errors = [], [], []

        # Setup progress bar
        iterator = tqdm(
            df.iterrows(),
            total=len(df),
            desc="Evaluating hair classification"
        ) if verbose else df.iterrows()

        # Run inference on each image
        for idx, row in iterator:
            start = time.time()
            pred = self.classifier.predict(
                encode_image(row["image_path"])
            )
            latency = time.time() - start

            preds.append(pred)
            times.append(latency)
            
            # Track errors for analysis
            if pred != row["type"]:
                errors.append({
                    "index": idx,
                    "image_path": row["image_path"],
                    "true_label": row["type"],
                    "predicted_label": pred,
                    "latency": latency
                })

            # Update progress bar with stats
            if verbose:
                iterator.set_postfix({
                    "last_pred": pred,
                    "accuracy": f"{accuracy_score(df['type'][:len(preds)], preds):.3f}",
                    "avg_latency": f"{np.mean(times):.2f}s"
                })

        # Add predictions to dataframe
        df["predicted_label"] = preds
        df["latency"] = times
        
        # Save results if requested
        if save_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"results_evaluation_{timestamp}.csv"
            df.to_csv(output_path, index=False)
            if verbose:
                print(f"\n✓ Results saved to: {output_path}")

        # Get unique labels for metrics
        labels = sorted(df["type"].unique())
        
        # Calculate comprehensive metrics
        y_true = df["type"]
        y_pred = df["predicted_label"]
        
        # Compute latency statistics
        latency_stats = {
            "mean": np.mean(times),
            "median": np.median(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "p25": np.percentile(times, 25),
            "p75": np.percentile(times, 75),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99)
        }
        
        # Generate classification report
        class_report = classification_report(
            y_true, y_pred, 
            labels=labels,
            output_dict=True,
            zero_division=0
        )
        
        # Create error analysis dataframe
        error_df = pd.DataFrame(errors) if errors else pd.DataFrame()
        
        return {
            # Accuracy metrics
            "accuracy": accuracy_score(y_true, y_pred),
            "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
            "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
            "weighted_precision": precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
            "weighted_recall": recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
            
            # Confusion matrix and detailed metrics
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
            "labels": labels,
            "classification_report": class_report,
            
            # Latency analysis
            "latency_stats": latency_stats,
            "latency_times": times,
            
            # Detailed results
            "results": df,
            "error_analysis": error_df,
            "n_samples": len(df),
            "n_errors": len(errors)
        }
