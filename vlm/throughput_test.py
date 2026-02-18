"""
Throughput Testing Script for Hair Classification API

Tests API performance under concurrent load to measure:
- Requests per second at different concurrency levels
- Latency under load
- Error rates
- System stability
"""

import time
import asyncio
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import EvalConfig
from src.client import AzureLLMClient
from src.classifier import HairClassifier
from src.utils import encode_image
from dotenv import load_dotenv
import os
import numpy as np
from tqdm import tqdm

load_dotenv()


class ThroughputTester:
    """Test API throughput with concurrent requests."""
    
    def __init__(self, classifier, dataset_path):
        """Initialize throughput tester.
        
        Args:
            classifier: HairClassifier instance
            dataset_path: Path to dataset CSV
        """
        self.classifier = classifier
        self.df = pd.read_csv(dataset_path)
    
    def _process_single_image(self, image_path, true_label):
        """Process a single image and return results.
        
        Args:
            image_path: Path to image file
            true_label: Ground truth label
            
        Returns:
            Dictionary with timing and prediction info
        """
        start_time = time.time()
        try:
            # Encode and classify
            image_b64 = encode_image(image_path)
            prediction = self.classifier.predict(image_b64)
            latency = time.time() - start_time
            
            return {
                "image_path": image_path,
                "true_label": true_label,
                "predicted_label": prediction,
                "latency": latency,
                "success": True,
                "error": None
            }
        except Exception as e:
            latency = time.time() - start_time
            return {
                "image_path": image_path,
                "true_label": true_label,
                "predicted_label": "Error",
                "latency": latency,
                "success": False,
                "error": str(e)
            }
    
    def test_throughput(self, concurrency=5, max_samples=None, repetitions=1):
        """Test throughput at specified concurrency level.
        
        Args:
            concurrency: Number of concurrent requests
            max_samples: Maximum number of samples to test
            repetitions: Number of times to repeat each image for load testing
            
        Returns:
            Dictionary with throughput metrics
        """
        print(f"\n{'='*70}")
        print(f"Testing Throughput at Concurrency Level: {concurrency}")
        print(f"{'='*70}")
        
        # Prepare sample list
        df_test = self.df.head(max_samples) if max_samples else self.df
        
        # Create task list (with repetitions for sustained load)
        tasks = []
        for _ in range(repetitions):
            for _, row in df_test.iterrows():
                tasks.append((row["image_path"], row["type"]))
        
        total_requests = len(tasks)
        print(f"Total requests to process: {total_requests}")
        print(f"Concurrency level: {concurrency}")
        print(f"\nStarting test...")
        
        # Execute with thread pool
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._process_single_image, img, label): (img, label)
                for img, label in tasks
            }
            
            # Collect results with progress bar
            with tqdm(total=total_requests, desc="Processing") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        df_results = pd.DataFrame(results)
        
        successful_requests = df_results["success"].sum()
        failed_requests = total_requests - successful_requests
        error_rate = failed_requests / total_requests
        
        # Latency statistics (only for successful requests)
        successful_latencies = df_results[df_results["success"]]["latency"]
        
        throughput_rps = total_requests / total_time
        
        metrics = {
            "concurrency": concurrency,
            "total_requests": total_requests,
            "total_time": total_time,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "error_rate": error_rate,
            "throughput_rps": throughput_rps,
            "throughput_rpm": throughput_rps * 60,
            "avg_latency": successful_latencies.mean() if len(successful_latencies) > 0 else 0,
            "median_latency": successful_latencies.median() if len(successful_latencies) > 0 else 0,
            "p95_latency": np.percentile(successful_latencies, 95) if len(successful_latencies) > 0 else 0,
            "p99_latency": np.percentile(successful_latencies, 99) if len(successful_latencies) > 0 else 0,
            "min_latency": successful_latencies.min() if len(successful_latencies) > 0 else 0,
            "max_latency": successful_latencies.max() if len(successful_latencies) > 0 else 0,
            "results_df": df_results
        }
        
        # Print results
        print(f"\n{'='*70}")
        print(f"Throughput Test Results (Concurrency: {concurrency})")
        print(f"{'='*70}")
        print(f"Total Time:              {total_time:.2f}s")
        print(f"Total Requests:          {total_requests}")
        print(f"Successful Requests:     {successful_requests}")
        print(f"Failed Requests:         {failed_requests}")
        print(f"Error Rate:              {error_rate*100:.2f}%")
        print(f"\nThroughput:")
        print(f"  Requests/Second:       {throughput_rps:.2f} req/s")
        print(f"  Requests/Minute:       {throughput_rps * 60:.2f} req/min")
        print(f"\nLatency (Successful Requests):")
        print(f"  Mean:                  {metrics['avg_latency']:.3f}s")
        print(f"  Median:                {metrics['median_latency']:.3f}s")
        print(f"  P95:                   {metrics['p95_latency']:.3f}s")
        print(f"  P99:                   {metrics['p99_latency']:.3f}s")
        print(f"  Min:                   {metrics['min_latency']:.3f}s")
        print(f"  Max:                   {metrics['max_latency']:.3f}s")
        
        return metrics
    
    def compare_concurrency_levels(self, concurrency_levels=[1, 5, 10, 20], 
                                   max_samples=50, repetitions=1):
        """Compare performance across multiple concurrency levels.
        
        Args:
            concurrency_levels: List of concurrency levels to test
            max_samples: Maximum samples per test
            repetitions: Repetitions per sample
            
        Returns:
            List of metrics dictionaries
        """
        all_metrics = []
        
        for concurrency in concurrency_levels:
            metrics = self.test_throughput(
                concurrency=concurrency,
                max_samples=max_samples,
                repetitions=repetitions
            )
            all_metrics.append(metrics)
            
            # Pause between tests to avoid rate limiting
            if concurrency != concurrency_levels[-1]:
                print(f"\nPausing 10 seconds before next test...")
                time.sleep(10)
        
        # Summary comparison
        print(f"\n{'='*70}")
        print("THROUGHPUT COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Concurrency':<12} {'Throughput (req/s)':<20} {'Avg Latency (s)':<18} {'Error Rate':<12}")
        print("-" * 70)
        
        for m in all_metrics:
            print(f"{m['concurrency']:<12} {m['throughput_rps']:<20.2f} "
                  f"{m['avg_latency']:<18.3f} {m['error_rate']*100:<12.2f}%")
        
        return all_metrics


def main():
    """Main execution function."""
    print("=" * 70)
    print(" " * 20 + "THROUGHPUT TESTING")
    print("=" * 70)
    
    # Load configuration
    cfg = EvalConfig("config/eval.yaml")
    
    # Load prompt
    with open(cfg.paths["prompt"], "r") as f:
        prompt = f.read()
    
    # Initialize client and classifier
    client = AzureLLMClient(
        api_key=os.getenv("AZUREOPENAI_API_KEY"),
        endpoint=os.getenv("AZUREOPENAI_API_ENDPOINT"),
        api_version=os.getenv("AZUREOPENAI_API_VERSION", "2024-02-15-preview")
    )
    
    classifier = HairClassifier(client, cfg, prompt)
    
    # Initialize tester
    tester = ThroughputTester(classifier, cfg.paths["dataset"])
    
    # Run throughput tests at different concurrency levels
    print("\nTesting throughput at multiple concurrency levels...")
    print("This will take some time.\n")
    
    # Test configuration
    CONCURRENCY_LEVELS = [1, 5, 10]  # Add 20 if you want to test higher load
    MAX_SAMPLES = 20  # Number of unique images to test
    REPETITIONS = 2   # How many times to repeat for sustained load
    
    results = tester.compare_concurrency_levels(
        concurrency_levels=CONCURRENCY_LEVELS,
        max_samples=MAX_SAMPLES,
        repetitions=REPETITIONS
    )
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"throughput_results_{timestamp}.csv"
    
    summary_data = []
    for m in results:
        summary_data.append({
            "concurrency": m["concurrency"],
            "total_requests": m["total_requests"],
            "total_time": m["total_time"],
            "throughput_rps": m["throughput_rps"],
            "throughput_rpm": m["throughput_rpm"],
            "avg_latency": m["avg_latency"],
            "median_latency": m["median_latency"],
            "p95_latency": m["p95_latency"],
            "p99_latency": m["p99_latency"],
            "error_rate": m["error_rate"],
            "successful_requests": m["successful_requests"],
            "failed_requests": m["failed_requests"]
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_file, index=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "=" * 70)
    print("THROUGHPUT TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
