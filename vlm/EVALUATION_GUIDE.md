# Hair Classification Performance Evaluation

Comprehensive evaluation and performance analysis for hair classification using Vision-Language Models (VLMs).

## 📋 Overview

This project evaluates VLM APIs for hair texture classification according to the Andre Walker hair typing system (1A-4C). The evaluation includes:

- **Dataset Analysis** - Class distribution, imbalances
- **Classification Metrics** - Accuracy, F1-score, precision, recall
- **Confusion Matrix** - Visual analysis of misclassifications
- **Latency Analysis** - Response time distribution and percentiles
- **Throughput Testing** - Performance under concurrent load
- **Error Analysis** - Detailed misclassification patterns

## 🗂️ Project Structure

```
├── run.ipynb                  # Main analysis notebook with visualizations
├── throughput_test.py         # Throughput testing script
├── PERFORMANCE_REPORT.md      # Report template with dataset info
├── src/
│   ├── evaluator.py          # Enhanced evaluator with comprehensive metrics
│   ├── classifier.py         # Hair classification logic
│   ├── client.py             # API client (Azure OpenAI)
│   ├── config.py             # Configuration management
│   └── utils.py              # Utility functions
├── config/
│   └── eval.yaml             # Evaluation configuration
├── data/
│   └── annotations.csv       # Dataset with 150 images
└── prompts/
    └── vision_only.txt       # System prompt for VLM
```

## 📊 Dataset Information

- **Total Images**: 150
- **Classes**: 10 hair types (1A, 1B, 1C, 2A, 2B, 2C, 3A, 3B, 4A, 4B)
- **Format**: Segmented hair masks (isolated hair on black background)

### Class Distribution

| Hair Type | Count | Percentage | Category |
|-----------|-------|------------|----------|
| 1A | 25 | 16.7% | Straight |
| 1B | 18 | 12.0% | Straight |
| 1C | 18 | 12.0% | Straight |
| 2A | 17 | 11.3% | Wavy |
| 2B | 8 | 5.3% | Wavy |
| 2C | 4 | 2.7% | Wavy (⚠️ underrepresented) |
| 3A | 14 | 9.3% | Curly |
| 3B | 16 | 10.7% | Curly |
| 4A | 10 | 6.7% | Kinky/Coily |
| 4B | 20 | 13.3% | Kinky/Coily |

**Class Imbalance**: Max/Min ratio of 6.25:1 (25:4)

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Activate virtual environment
source env/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### 2. Configure API Credentials

Create a `.env` file or set environment variables:

```bash
export AZUREOPENAI_API_KEY="your-api-key"
export AZUREOPENAI_API_ENDPOINT="your-endpoint"
export AZUREOPENAI_API_VERSION="2024-02-15-preview"
```

### 3. Configure Evaluation Settings

Edit `config/eval.yaml`:

```yaml
api:
  provider: azure_openai
  model: gpt-4o-mini
  temperature: 0
  timeout: 30
  retries: 2

evaluation:
  max_samples: null   # null = full dataset, or set a number for testing
```

## 📓 Running the Analysis Notebook

The `run.ipynb` notebook provides comprehensive analysis with visualizations.

### What the Notebook Does:

1. **Dataset Analysis**
   - Class distribution charts (bar plot, pie chart)
   - Imbalance statistics
   - Sample counts per class

2. **Model Evaluation**
   - Runs inference on all images
   - Calculates comprehensive metrics
   - Saves results to CSV with timestamp

3. **Performance Metrics**
   - Overall accuracy, weighted F1, macro F1
   - Per-class precision, recall, F1-score
   - Confusion matrix (absolute & normalized)

4. **Latency Analysis**
   - Mean, median, P95, P99 latency
   - Distribution plots (histogram, box plot)
   - Time series of latency over requests

5. **Error Analysis**
   - Most common misclassification patterns
   - Detailed error table
   - Misclassification heatmap

6. **Summary Report**
   - Comprehensive metrics summary
   - Best/worst performing classes
   - Throughput estimates

### Running the Notebook:

```bash
# Open in Jupyter or VS Code
jupyter notebook run.ipynb

# Or run in VS Code with Jupyter extension
```

**Note**: Set `max_samples` in `config/eval.yaml`:
- `max_samples: 10` - Quick test (10 images)
- `max_samples: 50` - Medium test (50 images)
- `max_samples: null` - Full dataset (150 images)

### Expected Runtime:

- **10 samples**: ~2-3 minutes
- **50 samples**: ~10-15 minutes
- **150 samples**: ~30-45 minutes

(Depends on API latency, typically 1-3 seconds per image)

## ⚡ Throughput Testing

Test API performance under concurrent load to measure scalability.

### Running Throughput Tests:

```bash
python throughput_test.py
```

### What It Tests:

- **Concurrency Levels**: 1, 5, 10 (configurable)
- **Metrics**:
  - Requests per second (RPS)
  - Requests per minute (RPM)
  - Latency under load (mean, median, P95, P99)
  - Error rate
  - Success rate

### Customizing Tests:

Edit variables in `throughput_test.py`:

```python
CONCURRENCY_LEVELS = [1, 5, 10, 20]  # Test these concurrency levels
MAX_SAMPLES = 20                      # Number of unique images
REPETITIONS = 2                       # Repeat each image N times
```

### Output:

- Console output with real-time progress
- CSV file: `throughput_results_YYYYMMDD_HHMMSS.csv`

## 📈 Metrics Included

### Classification Metrics:
- ✅ Overall Accuracy
- ✅ Weighted F1-Score (accounts for class imbalance)
- ✅ Macro F1-Score (average across classes)
- ✅ Per-class Precision, Recall, F1
- ✅ Confusion Matrix (absolute & normalized)

### Latency Metrics:
- ✅ Mean, Median latency
- ✅ P25, P75, P95, P99 percentiles
- ✅ Min, Max latency
- ✅ Standard deviation

### Throughput Metrics:
- ✅ Requests per second (RPS)
- ✅ Requests per minute (RPM)
- ✅ Latency under concurrent load
- ✅ Error rate at different concurrency levels

### Additional Analysis:
- ✅ Error patterns and misclassification pairs
- ✅ Best/worst performing classes
- ✅ Class imbalance impact

## 📊 Understanding the Results

### Accuracy Metrics:

- **Overall Accuracy**: Percentage of correct predictions (may be misleading with imbalanced data)
- **Weighted F1**: F1 weighted by class support (better for imbalanced data)
- **Macro F1**: Simple average of per-class F1 (treats all classes equally)

**Use Weighted F1 as the primary metric** for this imbalanced dataset.

### Confusion Matrix:

- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications
- Look for patterns (e.g., 1A confused with 1B = adjacent class confusion)

### Latency:

- **Mean**: Average response time
- **P95**: 95% of requests complete within this time
- **P99**: 99% of requests complete within this time (worst-case performance)

### Throughput:

- **Sequential (concurrency=1)**: Baseline throughput
- **Concurrent (concurrency>1)**: Tests if API can handle parallel requests
- Watch for latency increases at higher concurrency

## 🎯 Using Results for Your Report

### 1. Dataset Section:
- Use class distribution table and charts from notebook
- Mention imbalance ratio and strategies to address it
- Include dataset statistics from Section 1

### 2. Metrics Section:
- Copy accuracy metrics from Section 3
- Include confusion matrix images from Section 4
- Add per-class metrics table from Section 5

### 3. Latency Section:
- Use latency statistics from Section 6
- Include distribution plots (histogram, box plot)
- Highlight P95/P99 for SLA discussions

### 4. Throughput Section:
- Run `throughput_test.py` for different APIs
- Compare RPS across APIs
- Note error rates and latency degradation

### 5. Error Analysis:
- Document common misclassification patterns from Section 7
- Discuss why certain classes are confused
- Propose improvements

## 🔄 Comparing Multiple APIs

To compare different APIs (e.g., GPT-4o vs Gemini vs Claude):

1. **Update config**:
   ```yaml
   api:
     provider: google_gemini  # or anthropic_claude
     model: gemini-1.5-pro
   ```

2. **Create new client** in `src/client.py` for the API

3. **Run notebook** for each API

4. **Compare results**:
   - Accuracy metrics
   - Latency distributions
   - Throughput at same concurrency
   - Cost per 1000 requests

## 📝 Output Files

All results are automatically saved with timestamps:

- `results_evaluation_YYYYMMDD_HHMMSS.csv` - Detailed predictions from notebook
- `throughput_results_YYYYMMDD_HHMMSS.csv` - Throughput test summary

## 🛠️ Troubleshooting

### API Errors:
- Check API credentials in `.env`
- Verify API quota/rate limits
- Increase `timeout` in `config/eval.yaml`

### Out of Memory:
- Reduce `max_samples` in config
- Test with smaller batches

### Slow Performance:
- Check network connection
- Verify API region (latency varies by region)
- Monitor API status page

## 📚 Additional Metrics to Consider

For a comprehensive report, consider adding:

1. **Cost Analysis**:
   - Cost per 1000 requests
   - Cost per correct classification
   - Cost-accuracy tradeoff

2. **Top-K Accuracy**:
   - Top-3 accuracy (is correct label in top 3 predictions?)
   - Useful for recommendation systems

3. **Within-Category Accuracy**:
   - Accuracy within Straight (1A-1C)
   - Accuracy within Curly (3A-3B)
   - Helps understand fine-grained vs coarse classification

4. **Confidence Analysis**:
   - Parse "Reason" field from JSON responses
   - Correlate confidence with correctness

## 🎓 Codebase Documentation

All code is now comprehensively commented:

- [src/evaluator.py](src/evaluator.py) - Enhanced evaluator with detailed metrics
- [src/classifier.py](src/classifier.py) - Classification logic with retry handling
- [src/client.py](src/client.py) - API client implementation
- [src/utils.py](src/utils.py) - Utility functions

## 📞 Support

For questions or issues:
1. Check existing output files for clues
2. Review error messages in notebook cells
3. Verify API credentials and quotas

---

**Ready to start?** Open `run.ipynb` and run the cells sequentially! 🚀
