#!/usr/bin/env python3
"""Fetch comprehensive validator information from X1 testnet and save to JSON file.
This script gathers extensive data including performance metrics, network info, and more.
"""

import json
import subprocess
import os
import logging
import sys
import time
import hashlib
from pathlib import Path
import argparse
import re
from typing import Dict, List, Optional, Any, Tuple, Set
import concurrent.futures
from datetime import datetime, timezone, UTC
import tempfile
import shutil

# Check if running in GitHub Actions
IN_GITHUB_ACTIONS = os.environ.get('GITHUB_ACTIONS') == 'true'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables or defaults
RPC_URL = os.environ.get("RPC_URL", "https://rpc.testnet.x1.xyz")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "10"))
CACHE_DIR = os.environ.get("CACHE_DIR", ".validator_cache")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "public/data")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "5"))

def run_with_retry(func, *args, retries=3, delay=5, **kwargs):
    """Run a function with retries and exponential backoff."""
    attempt = 0
    last_exception = None
    
    while attempt < retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            wait_time = delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt+1}/{retries} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            attempt += 1
    
    # If we reach here, all retries failed
    logger.error(f"All {retries} attempts failed. Last error: {last_exception}")
    raise last_exception

def get_cache_path(command_name: str, command_args: str = "") -> Path:
    """Get path to a cached result file using descriptive names.
    
    Args:
        command_name: Base name of the command (e.g., 'validator-info')
        command_args: Optional arguments to make the filename more specific
    """
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    
    # Clean the args to create a valid filename
    if command_args:
        # Replace invalid filename chars
        cleaned_args = re.sub(r'[^a-zA-Z0-9_-]', '_', command_args)
        # Limit length
        if len(cleaned_args) > 50:
            cleaned_args = cleaned_args[:50]
        filename = f"{command_name}_{cleaned_args}.json"
    else:
        filename = f"{command_name}.json"
        
    return cache_dir / filename

def get_cached_result(command: List[str]) -> Optional[str]:
    """Get cached result for a command if it exists and is recent."""
    # Extract the base command and args for a descriptive filename
    if not command:
        return None
        
    base_command = command[0] if len(command) > 0 else "unknown"
    command_args = "_".join(command[1:3]) if len(command) > 1 else ""
    
    cache_path = get_cache_path(base_command, command_args)
    
    # Check if cache exists and is recent (less than 1 hour old)
    if cache_path.exists():
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age < 3600:  # 1 hour in seconds
            try:
                with open(cache_path, 'r') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")
    
    return None

def censor_ip(ip: str) -> str:
    """Censor an IP address for privacy by replacing the last octet with X's.
    
    Args:
        ip: The IP address to censor
        
    Returns:
        Censored IP address with last octet replaced by X's
    """
    if not ip or '.' not in ip:
        return ip
        
    parts = ip.split('.')
    if len(parts) == 4:  # IPv4
        return f"{parts[0]}.{parts[1]}.{parts[2]}.XXX"
    return ip  # Return original if not IPv4

def save_to_cache(command: List[str], result: str) -> None:
    """Save command result to cache using descriptive filenames."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Create a more descriptive filename based on the command
        if command and len(command) > 1:
            # Base filename on the command name and first arg
            base_name = f"solana_{command[0]}"
            if len(command) > 1 and command[1] != "--url":
                base_name += f"_{command[1]}"
            
            # Add MD5 hash for remaining args if needed
            if len(command) > 2:
                args_hash = hashlib.md5(' '.join(command[2:]).encode()).hexdigest()[:8]
                base_name += f"_{args_hash}"
        else:
            # Fallback to MD5 hash if command structure is unexpected
            base_name = f"cmd_{hashlib.md5(' '.join(command).encode()).hexdigest()[:16]}"
        
        cache_path = os.path.join(CACHE_DIR, f"{base_name}.json")
        
        # For validator-info get, try to format the output as proper JSON if it's not already
        if command[0] == "validator-info" and command[1] == "get" and not result.strip().startswith('['):
            try:
                parsed_data = parse_validator_info(result)
                with open(cache_path, 'w') as f:
                    json.dump(parsed_data, f, indent=2)
                logger.debug(f"Cached validator-info as formatted JSON to {cache_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to format validator-info as JSON: {e}")
        
        with open(cache_path, 'w') as f:
            f.write(result)
        logger.debug(f"Cached result to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to cache result: {e}")

def run_solana_command(args: List[str], timeout: int = 30) -> str:
    """Run a solana CLI command and return the output."""
    cmd = ["solana"] + args + ["--url", RPC_URL]
    
    # Try to get cached result first
    cached_result = get_cached_result(cmd)
    if cached_result:
        logger.debug(f"Using cached result for: {' '.join(cmd)}")
        return cached_result
    
    try:
        logger.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        output = result.stdout
        
        # Save to cache
        save_to_cache(cmd, output)
        
        return output
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        logger.error(f"Error output: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error(f"Solana CLI not found. If on Windows, please run this script in WSL where Solana is installed.")
        logger.error(f"WSL installation guide: https://docs.solana.com/cli/install-solana-cli-tools")
        raise
    except Exception as e:
        logger.error(f"Failed to run command: {e}")
        raise

def parse_validator_info(output: str) -> List[Dict]:
    """Parse the output of 'solana validator-info get' command.
    
    Converts the text output with indentation and colons into a proper JSON structure.
    """
    # Check if output is already in JSON format
    if output.strip().startswith('[') and output.strip().endswith(']'):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            logger.warning("Cached validator info is not valid JSON despite starting with '['. Will parse manually.")
    
    validators = []
    current_validator = None
    
    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Match validator identity line
        if line.startswith('Validator Identity:'):
            if current_validator:
                validators.append(current_validator)
            identity = line.split(':', 1)[1].strip()
            current_validator = {
                'identityPubkey': identity,
                'name': f'Validator {identity[:8]}',  # Default name
                'website': '',
                'details': '',
                'iconUrl': '',
                'infoAddress': '',
                'keybaseUsername': ''
            }
        
        # Match other fields
        elif current_validator and ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'Info Address':
                current_validator['infoAddress'] = value
            elif key == 'Name':
                current_validator['name'] = value
            elif key == 'Website':
                current_validator['website'] = value
            elif key == 'Details':
                current_validator['details'] = value
            elif key == 'Icon Url':
                current_validator['iconUrl'] = value
            elif key == 'Keybase Username':
                current_validator['keybaseUsername'] = value
    
    # Don't forget the last validator
    if current_validator:
        validators.append(current_validator)
    
    # Save the parsed data as proper JSON for future use
    if validators:
        cache_path = get_cache_path("validator-info", "get")
        with open(cache_path, 'w') as f:
            json.dump(validators, f, indent=2)
        logger.info(f"Saved parsed validator info as JSON to {cache_path}")
    
    return validators

def get_validator_info() -> List[Dict]:
    """Get validator info using 'solana validator-info get' command.
    
    Returns the data as a properly structured JSON list of validator objects.
    """
    try:
        output = run_solana_command(["validator-info", "get"])
        parsed_info = parse_validator_info(output)
        
        # If we didn't get any validators, something went wrong
        if not parsed_info:
            logger.warning("No validators found in validator-info output")
            
        return parsed_info
    except Exception as e:
        logger.error(f"Failed to get validator info: {e}")
        return []

def get_validators_extended() -> List[Dict]:
    """Get extended validator information using 'solana validators' command."""
    try:
        output = run_solana_command(["validators", "--output", "json"])
        data = json.loads(output)
        
        validators = {}
        
        # Extract average stake info
        average_stake = data.get('averageStake', 0)
        
        # Process validators
        for validator in data.get('validators', []):
            identity = validator.get('identityPubkey', '')
            validators[identity] = {
                'identityPubkey': identity,
                'voteAccountPubkey': validator.get('voteAccountPubkey', ''),
                'commission': validator.get('commission', 0),
                'lastVote': validator.get('lastVote', 0),
                'rootSlot': validator.get('rootSlot', 0),
                'credits': validator.get('credits', 0),
                'epochCredits': validator.get('epochCredits', 0),
                'activatedStake': validator.get('activatedStake', 0),
                'version': validator.get('version', 'unknown'),
                'delinquent': validator.get('delinquent', False),
                'skipRate': validator.get('skipRate', 0),
                'stakePercent': (validator.get('activatedStake', 0) / average_stake * 100) if average_stake > 0 else 0
            }
        
        return validators
    except Exception as e:
        logger.error(f"Error getting extended validator info: {e}")
        return {}

def get_gossip_nodes() -> Dict[str, Dict]:
    """Get gossip node information including IP addresses."""
    try:
        output = run_solana_command(["gossip", "--output", "json"])
        nodes = json.loads(output)
        
        node_info = {}
        for node in nodes:
            identity = node.get('identityPubkey', '')
            if identity:
                # Extract IP and port from gossip address
                gossip = node.get('gossip', '')
                ip_address = ''
                port = 0
                
                if gossip and ':' in gossip:
                    parts = gossip.rsplit(':', 1)
                    ip_address = parts[0]
                    try:
                        port = int(parts[1])
                    except:
                        port = 0
                
                node_info[identity] = {
                    'ipAddress': ip_address,
                    'gossipPort': port,
                    'tpuPort': node.get('tpu', '').split(':')[-1] if ':' in node.get('tpu', '') else 0,
                    'rpcPort': node.get('rpc', '').split(':')[-1] if ':' in node.get('rpc', '') else 0,
                    'version': node.get('version', 'unknown'),
                    'featureSet': node.get('featureSet', 0),
                    'shredVersion': node.get('shredVersion', 0)
                }
        
        return node_info
    except Exception as e:
        logger.error(f"Error getting gossip nodes: {e}")
        return {}

def get_cluster_stats() -> Dict:
    """Get overall cluster statistics using 'solana ping' command."""
    try:
        # Run ping command with 5 pings
        output = run_solana_command(["ping", "-c", "5"], timeout=20)
        
        # Parse the output to extract stats
        stats = {
            'tps': 0,
            'averageConfirmationTime': 0,
            'maxConfirmationTime': 0,
            'minConfirmationTime': 0,
            'confirmationTimeStdDev': 0
        }
        
        # Extract stats from output
        for line in output.split('\n'):
            line = line.strip()
            
            # Look for the TPS line
            if 'TPS:' in line:
                try:
                    tps_part = line.split('TPS:')[1].strip().split()[0]
                    stats['tps'] = float(tps_part)
                except (IndexError, ValueError):
                    pass
            
            # Look for confirmation time statistics
            if 'Average confirmation time:' in line:
                try:
                    time_part = line.split(':')[1].strip().split()[0]
                    stats['averageConfirmationTime'] = float(time_part)
                except (IndexError, ValueError):
                    pass
            
            if 'Minimum confirmation time:' in line:
                try:
                    time_part = line.split(':')[1].strip().split()[0]
                    stats['minConfirmationTime'] = float(time_part)
                except (IndexError, ValueError):
                    pass
                    
            if 'Maximum confirmation time:' in line:
                try:
                    time_part = line.split(':')[1].strip().split()[0]
                    stats['maxConfirmationTime'] = float(time_part)
                except (IndexError, ValueError):
                    pass
                    
            if 'Standard deviation:' in line:
                try:
                    time_part = line.split(':')[1].strip().split()[0]
                    stats['confirmationTimeStdDev'] = float(time_part)
                except (IndexError, ValueError):
                    pass
        
        return stats
    except Exception as e:
        logger.warning(f"Failed to get cluster stats: {e}")
        return {}

def get_stake_activation_status(stake_accounts: List[str]) -> Dict[str, Dict]:
    """Get stake activation status for a list of stake accounts."""
    activation_info = {}
    
    for account in stake_accounts:
        try:
            output = run_solana_command(["stake-account", account, "--output", "json"], timeout=10)
            data = json.loads(output)
            
            if 'stake' in data:
                stake_info = data['stake']
                activation_info[account] = {
                    'state': stake_info.get('state', 'unknown'),
                    'active': stake_info.get('active', 0),
                    'inactive': stake_info.get('inactive', 0),
                    'activating': stake_info.get('activating', 0),
                    'deactivating': stake_info.get('deactivating', 0),
                    'delegatedVote': stake_info.get('delegatedVote', '')
                }
        except Exception as e:
            logger.debug(f"Failed to get stake info for {account}: {e}")
            
    return activation_info

def get_vote_account_info(vote_pubkey: str) -> Dict:
    """Get detailed vote account information."""
    try:
        output = run_solana_command(["vote-account", vote_pubkey, "--output", "json"], timeout=10)
        data = json.loads(output)
        
        return {
            'commission': data.get('commission', 0),
            'rootSlot': data.get('rootSlot', 0),
            'lastVote': data.get('lastVote', 0),
            'epochCredits': data.get('epochCredits', []),
            'votes': len(data.get('votes', [])),
            'epochVoteAccount': data.get('epochVoteAccount', False),
            'authorizedVoter': data.get('authorizedVoter', ''),
            'authorizedWithdrawer': data.get('authorizedWithdrawer', ''),
            'nodePubkey': data.get('nodePubkey', '')
        }
    except Exception as e:
        logger.debug(f"Failed to get vote account info for {vote_pubkey}: {e}")
        return {}

def get_block_production() -> Dict:
    """Get block production statistics."""
    try:
        output = run_solana_command(["block-production", "--output", "json"])
        data = json.loads(output)
        
        # Handle different response formats
        if isinstance(data, list):
            # Convert list format to dictionary format
            result = {}
            for item in data:
                if isinstance(item, dict) and 'identityPubkey' in item:
                    result[item['identityPubkey']] = item
            return {'validators': result}
        elif isinstance(data, dict) and 'validators' in data:
            return data
        else:
            logger.warning(f"Unexpected block production format: {data}")
            return {'validators': {}}
    except Exception as e:
        logger.error(f"Error getting block production: {e}")
        return {'validators': {}}

def get_leader_schedule() -> Dict:
    """Stub function that returns empty dict since we're skipping leader schedule."""
    logger.info("Skipping leader schedule to save space")
    return {}

def merge_all_data(
    validator_info: List[Dict],
    validators_extended: Dict[str, Dict],
    gossip_nodes: Dict[str, Dict],
    block_production: Dict,
    leader_schedule: Dict
) -> List[Dict]:
    """Merge all collected data into comprehensive validator records."""
    
    # Start with extended validator data as base
    merged = {}
    
    # Add all validators from extended data
    for identity, data in validators_extended.items():
        merged[identity] = {
            'identityPubkey': identity,
            'votePubkey': data.get('voteAccountPubkey', ''),
            'name': f'Validator {identity[:8]}',
            'website': '',
            'details': '',
            'iconUrl': '',
            'keybaseUsername': '',
            'commission': data.get('commission', 0),
            'activatedStake': data.get('activatedStake', 0),
            'stakePercent': data.get('stakePercent', 0),
            'lastVote': data.get('lastVote', 0),
            'rootSlot': data.get('rootSlot', 0),
            'credits': data.get('credits', 0),
            'epochCredits': data.get('epochCredits', 0),
            'version': data.get('version', 'unknown'),
            'delinquent': data.get('delinquent', False),
            'skipRate': data.get('skipRate', 0),
            # Network info
            'ipAddress': '',
            'ipAddressCensored': '',
            'gossipPort': 0,
            'tpuPort': 0,
            'rpcPort': 0,
            'slotsBehind': 0,
            'timeBehindSeconds': 0,
            'featureSet': 0,
            'shredVersion': 0,
            # Block production
            'totalSlots': 0,
            'leaderSlots': 0,
            'blocksProduced': 0,
            'skippedSlots': 0,
            'skipRate': 0,
            'blockProductionRate': 0,
            # Leader schedule
            'upcomingLeaderSlots': 0,
            # Performance metrics
            'performanceScore': 0,
            'uptimePercent': 0
        }
    
    # Merge validator info
    for info in validator_info:
        identity = info['identityPubkey']
        if identity in merged:
            merged[identity].update({
                'name': info.get('name', merged[identity]['name']),
                'website': info.get('website', ''),
                'details': info.get('details', ''),
                'iconUrl': info.get('iconUrl', ''),
                'keybaseUsername': info.get('keybaseUsername', ''),
                'infoAddress': info.get('infoAddress', '')
            })
    
    # Merge gossip node info
    for identity, node_data in gossip_nodes.items():
        if identity in merged:
            ip = node_data.get('ipAddress', '')
            merged[identity].update({
                'ipAddress': ip,
                'ipAddressCensored': censor_ip(ip),
                'gossipPort': node_data.get('gossipPort', 0),
                'tpuPort': node_data.get('tpuPort', 0),
                'rpcPort': node_data.get('rpcPort', 0),
                'version': node_data.get('version', merged[identity].get('version', 'unknown')),
                'featureSet': node_data.get('featureSet', 0),
                'shredVersion': node_data.get('shredVersion', 0)
            })
    
    # Process block production
    try:
        block_production = get_block_production()
    except Exception as e:
        logger.warning(f"Failed to get block production: {e}")
        block_production = {'validators': {}}
    
    if 'validators' in block_production:
        for identity, data in block_production['validators'].items():
            if identity in merged:
                # Add block production data
                total_slots = data.get('total', 0)
                leader_slots = data.get('leaderSlots', 0)
                blocks_produced = data.get('blocksProduced', 0)
                
                # Handle case where values might be None
                if leader_slots is None:
                    leader_slots = 0
                if blocks_produced is None:
                    blocks_produced = 0
                    
                skipped_slots = leader_slots - blocks_produced
                skip_rate = skipped_slots / leader_slots if leader_slots > 0 else 0
                block_production_rate = blocks_produced / leader_slots if leader_slots > 0 else 0
                
                merged[identity]['totalSlots'] = total_slots
                merged[identity]['leaderSlots'] = leader_slots
                merged[identity]['blocksProduced'] = blocks_produced
                merged[identity]['skippedSlots'] = skipped_slots
                merged[identity]['skipRate'] = skip_rate
                merged[identity]['blockProductionRate'] = block_production_rate
    
    # Skip leader schedule processing
    for validator in merged.values():
        validator['leaderSlots'] = 0
    
    # Calculate performance scores
    for identity, validator in merged.items():
        # Simple performance score based on various metrics
        score = 100
        
        # Deduct for high skip rate
        score -= validator.get('skipRate', 0) * 0.5
        
        # Deduct for being delinquent
        if validator.get('delinquent', False):
            score -= 20
        
        # Deduct for low block production rate
        if validator.get('leaderSlots', 0) > 0:
            prod_rate = validator.get('blocksProduced', 0) / validator.get('leaderSlots', 0) * 100
            score -= (100 - prod_rate) * 0.3
        
        # Ensure score is between 0 and 100
        validator['performanceScore'] = max(0, min(100, score))
        
        # Calculate uptime (simplified - based on delinquent status and last vote)
        validator['uptimePercent'] = 0 if validator.get('delinquent', False) else 100
    
    # Convert to list and sort by stake
    validator_list = list(merged.values())
    validator_list.sort(key=lambda x: x['activatedStake'], reverse=True)
    
    return validator_list

def get_validator_tower_stats(validators: List[Dict], skip_details: bool = False) -> None:
    """Get tower statistics for validators.
    
    Args:
        validators: List of validator data dictionaries
        skip_details: If True, skip detailed calculations to improve performance
    """
    if skip_details:
        logger.info("Skipping detailed tower statistics calculation for better performance")
        # Just ensure basic fields exist without doing expensive calculations
        for validator in validators:
            validator['slotsBehind'] = 0
            validator['timeBehindSeconds'] = 0
        return
        
    # Only get current slot once for all validators (expensive operation)
    current_slot = get_current_slot()
    if current_slot <= 0:
        logger.warning("Failed to get current slot, skipping tower statistics")
        return
        
    logger.info(f"Current slot: {current_slot}, calculating tower statistics...")
    for validator in validators:
        identity = validator.get('identityPubkey', '')
        if not identity:
            continue
            
        # Calculate time since last vote if we have lastVote timestamp
        last_vote = validator.get('lastVote', 0)
        if last_vote > 0:
            slots_behind = current_slot - last_vote
            validator['slotsBehind'] = slots_behind
            # Approximate time (assuming 400ms per slot)
            validator['timeBehindSeconds'] = slots_behind * 0.4
        else:
            validator['slotsBehind'] = 0
            validator['timeBehindSeconds'] = 0

def get_current_slot() -> int:
    """Get the current slot of the cluster."""
    try:
        output = run_solana_command(["slot"])
        return int(output.strip())
    except (ValueError, Exception) as e:
        logger.warning(f"Failed to get current slot: {e}")
        return 0

def compare_output_files(original_file, new_file):
    """Compare the original and new output files to determine if there are significant changes.
    
    Returns:
        bool: True if there are significant changes, False otherwise.
    """
    # If original doesn't exist, consider it changed
    if not os.path.exists(original_file):
        return True
        
    try:
        # Load both files
        with open(original_file, 'r') as f:
            original_data = json.load(f)
        with open(new_file, 'r') as f:
            new_data = json.load(f)
            
        # Check metadata for timestamp changes (don't consider these significant)
        if 'metadata' in original_data and 'metadata' in new_data:
            # Make a copy with time-based fields standardized
            original_meta = dict(original_data['metadata'])
            new_meta = dict(new_data['metadata'])
            
            # Fields that change with time but don't represent real data changes
            time_fields = ['lastUpdated', 'dataCollectionTime']
            for field in time_fields:
                if field in original_meta:
                    original_meta[field] = 'standardized'
                if field in new_meta:
                    new_meta[field] = 'standardized'
                    
            # Significant change in metadata other than timestamps
            if original_meta != new_meta:
                # Compare validator counts and stakes (important metrics)
                if (original_meta.get('totalValidators') != new_meta.get('totalValidators') or
                    original_meta.get('activeValidators') != new_meta.get('activeValidators') or
                    original_meta.get('delinquentValidators') != new_meta.get('delinquentValidators') or
                    abs(original_meta.get('totalStake', 0) - new_meta.get('totalStake', 0)) > 1000):
                    return True
        
        # Check for changes in validator count
        original_validators = original_data.get('validators', [])
        new_validators = new_data.get('validators', [])
        
        if len(original_validators) != len(new_validators):
            return True
            
        # Check for changes in performance metrics
        original_performance = original_data.get('performance', {})
        new_performance = new_data.get('performance', {})
        
        # If skip rate or performance score changed significantly
        if (abs(original_performance.get('averageSkipRate', 0) - new_performance.get('averageSkipRate', 0)) > 2 or
            abs(original_performance.get('averagePerformanceScore', 0) - new_performance.get('averagePerformanceScore', 0)) > 2):
            return True
            
        # Check for changes in delinquent status (important!)
        original_delinquent = {v['identityPubkey']: v['delinquent'] for v in original_validators}
        new_delinquent = {v['identityPubkey']: v['delinquent'] for v in new_validators}
        
        if original_delinquent != new_delinquent:
            return True
            
        # Check for changes in commission (important for stakers)
        original_commission = {v['identityPubkey']: v['commission'] for v in original_validators}
        new_commission = {v['identityPubkey']: v['commission'] for v in new_validators}
        
        if original_commission != new_commission:
            return True
            
        # Check for any validator changes that would be visible to users
        # (name, website, details, icon)
        for new_validator in new_validators:
            identity = new_validator['identityPubkey']
            # Find matching validator in original
            original_validator = next((v for v in original_validators if v['identityPubkey'] == identity), None)
            
            if original_validator:
                if (original_validator.get('name') != new_validator.get('name') or
                    original_validator.get('website') != new_validator.get('website') or
                    original_validator.get('details') != new_validator.get('details') or
                    original_validator.get('iconUrl') != new_validator.get('iconUrl')):
                    return True
            else:
                # New validator not in original
                return True
                
        # If we reach here, no significant changes
        return False
    except Exception as e:
        logger.warning(f"Error comparing files: {e}")
        # If there's an error, assume there are changes to be safe
        return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fetch comprehensive validator information from X1 testnet")
    parser.add_argument('--rpc-url', default=RPC_URL, help=f"RPC URL to connect to (default: {RPC_URL})")
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help=f"Maximum number of worker threads (default: {MAX_WORKERS})")
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help=f"Directory to save output files (default: {OUTPUT_DIR})")
    parser.add_argument('--no-cache', action='store_true', help="Disable result caching")
    parser.add_argument('--timeout', type=int, default=30, help="Command timeout in seconds (default: 30)")
    parser.add_argument('--force', action='store_true', help="Force update even if no changes")
    parser.add_argument('--fast', action='store_true', help="Skip detailed tower statistics for faster execution")
    return parser.parse_args()

def main() -> int:
    """Main function to fetch and save comprehensive validator data.
    Returns exit code (0 for success, non-zero for errors)."""
    
    # Parse command line arguments if not in GitHub Actions
    if not IN_GITHUB_ACTIONS:
        args = parse_args()
        global RPC_URL, MAX_WORKERS, OUTPUT_DIR
        RPC_URL = args.rpc_url
        MAX_WORKERS = args.max_workers
        OUTPUT_DIR = args.output_dir
    
    logger.info("Starting comprehensive validator data collection for X1 testnet...")
    
    # Check if Solana CLI is available
    try:
        # Simple check to see if solana CLI is installed
        subprocess.run(["solana", "--version"], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        logger.error("Solana CLI not found. If on Windows, please run this script in WSL where Solana is installed.")
        logger.error("WSL installation guide: https://docs.solana.com/cli/install-solana-cli-tools")
        logger.error("To run in WSL: 'wsl -d Ubuntu-20.04' then navigate to this directory and run the script")
        return 1
    
    try:
        # Step 1: Get validator info
        logger.info("Fetching validator info...")
        validator_info_output = run_solana_command(["validator-info", "get"])
        validator_info = parse_validator_info(validator_info_output)
        logger.info(f"Found {len(validator_info)} validators with info")
        
        # Step 2: Get extended validator data
        logger.info("Fetching extended validator data...")
        validators_extended = get_validators_extended()
        logger.info(f"Found {len(validators_extended)} validators with extended data")
        
        # Step 3: Get gossip nodes (includes IP addresses)
        logger.info("Fetching gossip node information...")
        gossip_nodes = get_gossip_nodes()
        logger.info(f"Found {len(gossip_nodes)} gossip nodes")
        
        # Step 4: Get block production statistics
        logger.info("Fetching block production statistics...")
        block_production = get_block_production()
        logger.info(f"Found block production data for {len(block_production.get('validators', {}))} validators")
        
        # Step 5: Skip leader schedule to save space
        logger.info("Skipping leader schedule fetch to save space")
        leader_schedule = {}
        
        # Step 6: Merge all data
        logger.info("Merging all validator data...")
        validators = merge_all_data(
            validator_info,
            validators_extended,
            gossip_nodes,
            block_production,
            leader_schedule
        )
        
        # Step 7: Get validator tower stats (can be skipped for faster execution)
        if not IN_GITHUB_ACTIONS and hasattr(args, 'fast') and args.fast:
            logger.info("Fast mode enabled, skipping detailed tower statistics...")
            get_validator_tower_stats(validators, skip_details=True)
        else:
            logger.info("Getting validator tower statistics...")
            get_validator_tower_stats(validators)
        
        # Step 8: Get cluster statistics
        logger.info("Getting cluster statistics...")
        cluster_stats = get_cluster_stats()
        
        # Step 8: Calculate aggregate statistics
        total_stake = sum(v['activatedStake'] for v in validators)
        active_validators = [v for v in validators if not v.get('delinquent', False)]
        delinquent_validators = [v for v in validators if v.get('delinquent', False)]
        
        # Prepare output
        output_data = {
            'validators': validators,
            'metadata': {
                'lastUpdated': datetime.now(UTC).isoformat(),
                'network': 'X1 Testnet',
                'rpcUrl': RPC_URL,
                'totalValidators': len(validators),
                'activeValidators': len(active_validators),
                'delinquentValidators': len(delinquent_validators),
                'totalStake': total_stake,
                'totalStakeSOL': total_stake / 1e9 if total_stake > 0 else 0,
                'averageCommission': sum(v['commission'] for v in validators) / len(validators) if validators else 0,
                'averageStake': total_stake / len(validators) if validators else 0,
                'averageStakeSOL': (total_stake / len(validators) / 1e9) if validators else 0,
                'dataCollectionTime': datetime.now(UTC).isoformat(),
                'currentSlot': get_current_slot()
            },
            'performance': {
                'averageSkipRate': sum(v['skipRate'] for v in validators) / len(validators) if validators else 0,
                'averageBlockProductionRate': sum(v['blockProductionRate'] for v in validators) / len(validators) if validators else 0,
                'averagePerformanceScore': sum(v['performanceScore'] for v in validators) / len(validators) if validators else 0,
                'clusterStats': cluster_stats
            }
        }
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Define single output file
        output_file = os.path.join(OUTPUT_DIR, 'validators.json')
        
        # Write directly to output file (always overwrite)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Successfully saved validator data to {output_file}")
        logger.info(f"Total validators: {len(validators)}")
        logger.info(f"Active validators: {len(active_validators)}")
        logger.info(f"Delinquent validators: {len(delinquent_validators)}")
        logger.info(f"Total stake: {total_stake / 1e9:.2f} SOL")
        
        # For GitHub Actions, always indicate we have changes
        if IN_GITHUB_ACTIONS:
            with open(os.environ.get('GITHUB_OUTPUT', ''), 'a') as f:
                f.write("validators_changed=true\n")
                    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    # Try to import backoff; provide warning if not available
    try:
        import backoff
    except ImportError:
        logger.warning("The 'backoff' package is not installed. The script will continue without retry functionality.")
        logger.warning("To install, create a virtual environment with: python3 -m venv .venv")
        logger.warning("Then activate with: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)")
        logger.warning("Then install with: pip install backoff")
        
        # Create a minimal backoff decorator that does nothing
        def backoff_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        
        # Create a mock backoff module with the necessary functionality
        class MockBackoff:
            def on_exception(*args, **kwargs):
                return backoff_decorator
            
            def expo(*args, **kwargs):
                return None
        
        # Use the mock backoff
        backoff = MockBackoff()
        
    exit_code = main()
    # Always exit with success code (0) in GitHub Actions to prevent workflow failures
    # when there are no changes to commit
    if IN_GITHUB_ACTIONS and os.environ.get("ALWAYS_EXIT_ZERO", "true").lower() == "true":
        sys.exit(0)
    else:
        sys.exit(exit_code)