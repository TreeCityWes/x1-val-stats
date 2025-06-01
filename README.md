# X1 Validator Statistics

This repository contains a script that fetches comprehensive validator data from the Solana X1 testnet using Solana CLI.

## Features

- Fetches validator information from X1 testnet
- Collects metrics including stake, performance, block production, and more
- Outputs consolidated JSON data for analysis
- Automatically updates data via GitHub Actions

## Requirements

- Python 3.8+
- Solana CLI (v1.18.4+)

## Usage

Run the script with:

```bash
# Normal mode
python fetch_validators.py --rpc-url https://rpc.testnet.x1.xyz

# Fast mode (skips detailed tower statistics)
python fetch_validators.py --rpc-url https://rpc.testnet.x1.xyz --fast
```

### Options

- `--rpc-url`: Specify RPC endpoint (default: https://rpc.testnet.x1.xyz)
- `--cache-dir`: Cache directory for command outputs (default: .validator_cache)
- `--output-dir`: Output directory for JSON files (default: public/data)
- `--fast`: Skip detailed tower statistics for faster execution

## Automated Updates

This repository includes a GitHub Actions workflow that automatically updates the validator data every 6 hours.

## Output

The script generates a `validators.json` file containing detailed information about all validators on the X1 testnet.

## Environment Variables

- `RPC_URL`: Override the default RPC endpoint
- `MAX_WORKERS`: Limit parallel execution (default: 10)
- `MAX_RETRIES`: Number of command retries (default: 3)
- `CACHE_DIR`: Override cache directory
- `OUTPUT_DIR`: Override output directory
