# Unified Results Script

This script combines all agent result files from the LongBench evaluation into a single comprehensive JSON file.

## Features

- Unifies results from all agent JSONL files in the `results/` directory
- Stores metadata fields (question, choices, etc.) only once per entry
- Includes agent-specific data (prediction, correctness, token usage) for each agent
- Tracks and reports error counts for each agent
- Handles missing or malformed result files gracefully

## Usage

```bash
# Basic usage (processes all agent files in results/ directory)
python3 unified_results.py

# Specify a custom results directory
python3 unified_results.py --results_dir path/to/results

# Specify a custom output file
python3 unified_results.py --output path/to/output.json

# Process only specific agents
python3 unified_results.py --agents oneshot cot
```

## Output Format

The unified results are stored in a JSON file with the following structure:

```json
{
  "metadata": {
    "agents": ["oneshot", "cot", "quotes", "quotes_chunked"],
    "total_items": 503,
    "error_counts": {
      "oneshot": 191,
      "cot": 191,
      "quotes": 191
    }
  },
  "results": [
    {
      "_id": "unique-question-id",
      "domain": "Multi-Document QA",
      "sub_domain": "Legal",
      "difficulty": "hard",
      "length": "short",
      "question": "What do these two cases have in common?",
      "choice_A": "...",
      "choice_B": "...",
      "choice_C": "...",
      "choice_D": "...",
      "context": "...",
      "oneshot": {
        "error": "",
        "pred": "A",
        "judge": false,
        "token_usage": {
          "gpt-4o-mini-2024-07-18": {
            "prompt_tokens": 10695,
            "completion_tokens": 26,
            "total_tokens": 10721
          }
        }
      },
      "cot": {
        "error": "",
        "pred": "A",
        "judge": false,
        "token_usage": {
          // Token usage details
        }
      },
      // Other agent results...
    },
    // More results...
  ]
}
```

## Requirements

- Python 3.6+
- No additional dependencies required