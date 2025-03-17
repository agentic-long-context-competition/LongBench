# LongBench2 Competition Framework

This repository contains a flexible framework for evaluating multiple agent strategies on the LongBench v2 dataset (evaluation benchmark for long-context understanding, see [here](https://longbench2.github.io/)), keeping track of total token usage. The idea is to fix the model (for now, to 4o-mini) and see how good one can make this model perform just by improved scaffolding. This is to let different approaches stay comparable as new models get released.

To submit an entry, submit a PR according to the rules below.

The performance of the current sample agents is summarized in `results/result.txt`. The 191 errors are API errors, they come from the fact that they don't try to subdivide texts in anyway, so longer examples do not fit into the context window of 4o-mini. `python result.py` generates such summary statistics from the result files in `results/`.

RULES:
 - only 4o-mini and embeddings for now (scope is to compare different agentic scaffolding, not different foundation models or effect of finetuning). If there is interest, I may add a smarter model where the tokens are weighted accordingly (e.g. 4o where tokens count 15x as much as 4o-mini tokens, which reflects the pricing ratio)
 - allowed: just implementing a research paper, slightly modifying other submission
 - source code must become accessible in the repository (please submit a PR)
 - an agent can use at most 10 times as many total tokens for each entry as a one-shot query would use (see results/oneshot.jsonl for token statistics)
 - no hardcoding answers or similar up to my discretion (duh)

To submit an entry,

 - Add a new agent as below. IMPORTANT: Only use the API client passed to the run method. This enforces the allowed models (and sets a default model), tracks token usage, limits concurrency, implements backoff logic...
 - for the final evaluation, use `python main.py --max_concurrent 50 --agent cot --delete-old --logging --max_entries -1`,
 - run `python result.py` to generate summary statistics
 - (optional for now) upload your results (logs and results) to huggingface
 - submit a PR with your results

The file `specified_id_orderings.py` contains two orderings of the dataset IDs -- by context length in characters, and a random shuffling. These serve as Schelling points: If you only want to evaluate some of the entries for cost reasons, your results and those of other people will still be comparable.

## Features

- Multi-agent support with easy-to-extend registry
- Parallel processing of benchmark items
- Agent-specific results and comprehensive analytics
- Progress tracking with accuracy metrics

## Getting Started

### Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Put an `OPENAI_API_KEY` in .env or the environment variables.

### Running Evaluation

For testing an agent (running only on the shortest 50 entries by default):

```bash
python main.py --agent cot --max_concurrent 20 # change max_concurrent according to your resources
```

To run the final evaluation (on all 503 entries) with the default agent:

```bash
python main.py --max_concurrent 20 --agent oneshot --delete-old --logging --max_entries -1
```

Available agents:
- `oneshot`: Simple one-shot prompting agent
- `cot`: Chain-of-thought prompting agent that encourages step-by-step reasoning
- `quotes`: Quote extraction agent that focuses on finding supporting evidence

Additional parameters:
- `--max_entries`: Number of entries to process (default: 50, use -1 for all)
- `--max_concurrent`: Maximum concurrent API requests (default: 5)
- `--save_dir`: Directory to save results (default: "results") 
- `--processing_order`: Order to process items from `specified_id_orderings.py` - 'shuffled' or 'context_length' (measured in characters). (default: "context_length"). 
- `--logging`: Enable logging of API requests and responses
- `--delete-old`: Delete existing output files if they exist

### Processing Results

To process the results and generate statistics:

```bash
python result.py
```

Additional parameters:
- `--dir`: Directory containing result files (default: "results")
- `--output`: Output file path (default: "result.txt")
- `--compensated`: Use compensated accuracy calculation (25 % whenever no answer was given)

## Adding New Agents

To add a new agent:

1. Create a new file in the `agents` directory (e.g., `agents/my_agent.py`). You may copy and modify `agents/oneshot.py`.
2. Implement an agent class following the `AgentProtocol`
3. Import and register the agent in `agents/__init__.py`

## Original LongBench v2 Information

LongBench v2 is designed to assess the ability of LLMs to handle long-context problems requiring deep understanding and reasoning across real-world multitasks. Features include:
- **Length**: Context length ranging from 8k to 2M words
- **Difficulty**: Challenging even for human experts
- **Coverage**: Various realistic scenarios
- **Reliability**: Multiple-choice format for reliable evaluation

For more details about LongBench v2, visit the [official project page](https://longbench2.github.io).

## License

This project is licensed according to the original LongBench license.