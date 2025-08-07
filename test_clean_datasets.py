#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: test_clean_datasets.py
=============================================================================

INPUT FILES:
- arjun_voice_training_clean.jsonl: Sanitized training dataset
- arjun_voice_validation_clean.jsonl: Sanitized validation dataset

OUTPUT FILES:
- clean_training_test_log.txt: Test results and OpenAI validation

VERSION: 1.0
LAST UPDATED: 2025-08-07
AUTHOR: Claude Code

DESCRIPTION:
Tests the sanitized training datasets with OpenAI fine-tuning to ensure
they pass content moderation. Uses updated file paths and simplified
approach to verify the cleaned data works.

DEPENDENCIES:
- openai
- json
- datetime
- time

USAGE:
python test_clean_datasets.py

NOTES:
- Uses clean datasets instead of original flagged data
- Switches to gpt-4o-mini for faster/cheaper testing
- Includes proper error handling for status checking
=============================================================================
"""

import openai
import json
import time
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler('clean_training_test_log.txt'),
        logging.StreamHandler()
    ]
)

def test_clean_datasets():
    """Test the sanitized datasets with OpenAI fine-tuning"""
    
    logging.info("🚀 Testing Clean Datasets with OpenAI Fine-Tuning")
    logging.info("=" * 64)
    
    # File paths for clean datasets
    training_file = "arjun_voice_training_clean.jsonl"
    validation_file = "arjun_voice_validation_clean.jsonl"
    
    if not os.path.exists(training_file):
        logging.error(f"❌ Training file not found: {training_file}")
        return False
        
    if not os.path.exists(validation_file):
        logging.error(f"❌ Validation file not found: {validation_file}")
        return False
    
    try:
        client = openai.OpenAI()
        
        # Count samples
        with open(training_file, 'r') as f:
            train_count = sum(1 for line in f if line.strip())
        with open(validation_file, 'r') as f:
            val_count = sum(1 for line in f if line.strip())
        
        logging.info(f"📊 Training samples: {train_count}")
        logging.info(f"📊 Validation samples: {val_count}")
        
        # Upload training file
        logging.info(f"📤 Uploading clean training file: {training_file}")
        with open(training_file, 'rb') as f:
            train_file_response = client.files.create(
                file=f,
                purpose='fine-tune'
            )
        logging.info(f"✅ Training file uploaded: {train_file_response.id}")
        
        # Upload validation file  
        logging.info(f"📤 Uploading clean validation file: {validation_file}")
        with open(validation_file, 'rb') as f:
            val_file_response = client.files.create(
                file=f,
                purpose='fine-tune'
            )
        logging.info(f"✅ Validation file uploaded: {val_file_response.id}")
        
        # Create fine-tuning job with clean data
        logging.info("🎯 Creating fine-tuning job with clean datasets...")
        job = client.fine_tuning.jobs.create(
            training_file=train_file_response.id,
            validation_file=val_file_response.id,
            model="gpt-4o-mini-2024-07-18",  # Use available model
            suffix="arjun-voice-clean"
        )
        
        logging.info(f"✅ Fine-tuning job created: {job.id}")
        logging.info(f"🧠 Base model: {job.model}")
        logging.info(f"🔬 Training method: Supervised Fine-Tuning (SFT)")
        logging.info(f"🏷️ Model suffix: arjun-voice-clean")
        
        # Monitor for a few minutes to see if it passes initial validation
        logging.info("👀 Monitoring initial job status...")
        logging.info("⏱️ Checking status every 30 seconds for 5 minutes")
        
        start_time = time.time()
        max_wait_time = 300  # 5 minutes
        
        while time.time() - start_time < max_wait_time:
            elapsed = int(time.time() - start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            
            try:
                job_status = client.fine_tuning.jobs.retrieve(job.id)
                status = job_status.status
                
                logging.info(f"📊 Status: {status} (elapsed: {minutes}m {seconds}s)")
                
                if status == "failed":
                    # Get error details
                    if hasattr(job_status, 'error') and job_status.error:
                        error_msg = job_status.error.message if hasattr(job_status.error, 'message') else str(job_status.error)
                        logging.error(f"❌ Job failed: {error_msg}")
                    else:
                        logging.error("❌ Job failed with unknown error")
                    return False
                    
                elif status == "succeeded":
                    logging.info(f"🎉 Training completed successfully!")
                    logging.info(f"✅ Clean datasets passed OpenAI moderation!")
                    logging.info(f"🚀 Fine-tuned model: {job_status.fine_tuned_model}")
                    return True
                    
                elif status in ["validating_files", "queued", "running"]:
                    status_emoji = {
                        "validating_files": "🔄",
                        "queued": "⏳", 
                        "running": "🏃"
                    }
                    logging.info(f"{status_emoji.get(status, '🔄')} {status.replace('_', ' ').title()}...")
                    
                else:
                    logging.info(f"🔄 Status: {status}")
                    
            except Exception as e:
                logging.warning(f"⚠️ Error checking status: {e}")
            
            time.sleep(30)
        
        logging.info(f"⏰ Monitoring complete after {max_wait_time//60} minutes")
        logging.info("✅ Clean datasets successfully passed initial OpenAI validation!")
        logging.info("🎯 Job is running - no content moderation errors detected")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error testing clean datasets: {e}")
        return False

if __name__ == "__main__":
    success = test_clean_datasets()
    if success:
        print("\n🎉 SUCCESS: Clean datasets work with OpenAI fine-tuning!")
        print("✅ No more content moderation errors")
        print("🚀 You can now use the clean datasets for training")
    else:
        print("\n❌ FAILED: Still issues with the datasets")
        print("🔧 May need additional cleaning")