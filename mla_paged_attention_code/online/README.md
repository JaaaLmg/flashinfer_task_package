# Online Evaluation Results

This directory contains online evaluation results from the competition judge.

## How to Use

After submitting a kernel to the online judge, save the complete result output here.

### File Naming Convention:
- `baseline_result.txt` - First baseline submission
- `optimized_v1_result.txt` - First optimized version submission
- `optimized_v2_result.txt` - Second optimized version submission
- etc.

### What to Save:
Copy the **complete output** from the online judge, including:
- All test case results
- Pass/fail status
- Latency measurements
- Final score
- Any error messages

### Example Result File Format:
```
Case 1: PASSED (12.5ms)
Case 2: PASSED (25.3ms)
...
Final Score: 85.3
Rank: 42/150
```

## Current Status

### Waiting for submission:
- `optimized_v1_result.txt` - Submit `mla_paged_optimized.cu` and paste result here

Once results are saved, the optimization workflow will:
1. Calibrate local benchmarks to online scoring
2. Identify highest-value optimization opportunities
3. Generate Stage B candidates with predicted online impact
