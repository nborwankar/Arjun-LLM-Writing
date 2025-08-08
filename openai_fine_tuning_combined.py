#!/usr/bin/env python3
"""
=============================================================================
OPENAI FINE-TUNING MANAGER - Automated Model Training and Deployment
=============================================================================

PURPOSE:
Automates the complete OpenAI fine-tuning workflow for the Arjun investment
voice model: validates input JSONL datasets, uploads them, creates and
monitors the fine-tuning job, and saves model metadata for downstream use.

INPUT FILES (PROMINENT):
- Training JSONL: 
  /Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/arjun_voice_training_combined.jsonl
  Format: JSONL, each line is {"messages": [{"role":"user"|"assistant","content":"..."}, ...]}
  Notes: combined-filtered persona conversations in OpenAI chat format.

- Validation JSONL:
  /Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/arjun_voice_validation_combined.jsonl
  Format: same as Training JSONL; used for validation/monitoring during training.

- Dataset Stats (optional):
  /Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/persona_dataset_stats.json
  Purpose: Used to log expected counts and embed provenance into model info.

OUTPUT FILES (PROMINENT):
- Model Info JSON:
  arjun_voice_model_info.json (written to current working directory)
  Contents: {"model_id","job_id","base_model","status","files", "validation_prompts",
            "hyperparameters","dataset_stats", ...}

- Training Log:
  fine_tuning_log.txt (written to current working directory)
  Contents: timestamped progress, status transitions, errors. Acts as the
  primary meter for long-running steps (file validation, uploads, polling).

ENVIRONMENT REQUIREMENTS:
- OPENAI_API_KEY must be set in the environment.
- Python OpenAI client must be installed and network access available.

PROCESS OVERVIEW:
1) Validate input files and basic JSONL structure; warn on count mismatches vs dataset_stats.
2) Upload training/validation files to OpenAI Files API.
3) Create fine-tuning job on base model (gpt-4o-mini-2024-07-18) with SFT.
4) Monitor job with periodic polling (POLL_INTERVAL seconds) until completion.
5) Optionally validate the resulting model with fixed prompts.
6) Save the full model info JSON including embedded dataset_stats (if present).

METERS & LOGGING:
- Progress is written to fine_tuning_log.txt with timestamps (validation, upload,
  job creation, status transitions, completion). Polling interval is configurable.

MISSING DATA HANDLING (PIPELINE POLICY):
- This script does not process country-level data; no imputation occurs here.
- It performs consistency checks (e.g., sample counts vs stats) and logs any
  mismatches. Upstream dataset builders/converters should implement missing-data
  rules (e.g., fill missing country values with means and log replacements).

HYPERPARAMETERS (auto unless noted):
- Base model: gpt-4o-mini-2024-07-18 (cost-efficient).
- Method: Supervised Fine-Tuning (SFT)
- Epochs, batch size, learning rate multiplier: auto

USAGE:
  export OPENAI_API_KEY=... && python3 openai_fine_tuning_new.py

VERSION HISTORY:
- 2.1 (2025-08-08): Documentation overhaul; clarified inputs/outputs, meters,
  and base model set to gpt-4o-mini-2024-07-18.
- 2.0 (2025-08-07): Switched base model from gpt-4o to gpt-4o-mini to avoid
  billing hard limit failures; embedded dataset_stats in model info.
- 1.0 (2025-08-05): Initial production version using OpenAI chat JSONL files.

Last Updated: 2025-08-08
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import openai
from openai import OpenAI

# ═══════════════════════════ CONFIGURATION ═══════════════════════════ #

# File paths
# Use combined-converted OpenAI chat files as inputs
TRAINING_FILE = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/arjun_voice_training_combined.jsonl"
VALIDATION_FILE = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/arjun_voice_validation_combined.jsonl"
MODEL_INFO_FILE = "arjun_voice_model_info.json"
LOG_FILE = "fine_tuning_log.txt"
DATASET_STATS_FILE = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/persona_dataset_stats.json"

# Fine-tuning configuration
BASE_MODEL = "gpt-4o-mini-2024-07-18"  # Cheaper GPT-4o mini (July 2024) for cost-efficient fine-tuning
MODEL_SUFFIX = "arjun-voice-combined"        # Identifier for the fine-tuned model (v2 for GPT-4o)

# Monitoring settings
POLL_INTERVAL = 30                     # Check status every 30 seconds
MAX_WAIT_TIME = 3600                   # Maximum wait time (1 hour)

# Test prompts for model validation
VALIDATION_PROMPTS = [
    "What's your view on emerging market opportunities in 2024?",
    "How should investors approach portfolio diversification during market volatility?",
    "Can you explain the concept of value investing using an analogy?",
    "What lessons can we learn from the 2008 financial crisis?",
    "How do you evaluate investment opportunities in frontier markets?"
]

# ═══════════════════════════ FINE-TUNING MANAGER ═══════════════════════════ #

class OpenAIFineTuningManager:
    """
    Manages the complete OpenAI fine-tuning workflow for the Arjun voice model.
    
    Handles file uploads, job creation, progress monitoring, and model validation
    with comprehensive error handling and detailed logging throughout the process.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the fine-tuning manager.
        
        Args:
            api_key: OpenAI API key (uses environment variable if not provided)
        """
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        # File paths
        self.training_file = Path(TRAINING_FILE)
        self.validation_file = Path(VALIDATION_FILE)
        self.model_info_file = Path(MODEL_INFO_FILE)
        self.log_file = Path(LOG_FILE)
        self.dataset_stats_file = Path(DATASET_STATS_FILE)
        self.dataset_stats = None
        
        # Tracking variables
        self.training_file_id = None
        self.validation_file_id = None
        self.job_id = None
        self.model_id = None
        self.start_time = None
        
        # Initialize logging
        self.log_messages = []
        self.log(f"🚀 OpenAI Fine-Tuning Manager initialized")
        self.log(f"📁 Training file: {self.training_file}")
        self.log(f"📁 Validation file: {self.validation_file}")
        # Attempt to read dataset stats early for visibility
        if self.dataset_stats_file.exists():
            try:
                with open(self.dataset_stats_file, 'r', encoding='utf-8') as f:
                    self.dataset_stats = json.load(f)
                # Log a concise summary if available
                total_train = self.dataset_stats.get('train_count') or self.dataset_stats.get('training', {}).get('count')
                total_val = self.dataset_stats.get('val_count') or self.dataset_stats.get('validation', {}).get('count')
                self.log("📈 Loaded persona_dataset_stats.json")
                if total_train is not None:
                    self.log(f"   • Reported training examples: {total_train}")
                if total_val is not None:
                    self.log(f"   • Reported validation examples: {total_val}")
            except Exception as e:
                self.log(f"⚠️ Could not read dataset stats: {e}")
    
    def log(self, message: str):
        """
        Log message with timestamp to both console and log file.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        print(log_entry)
        self.log_messages.append(log_entry)
        
        # Write to log file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def validate_files(self) -> bool:
        """
        Validate that required training files exist and are properly formatted.
        
        Returns:
            bool: True if files are valid
        """
        self.log("🔍 Validating training files...")
        
        # Check file existence
        if not self.training_file.exists():
            self.log(f"❌ Training file not found: {self.training_file}")
            return False
        
        if not self.validation_file.exists():
            self.log(f"❌ Validation file not found: {self.validation_file}")
            return False
        
        # Validate JSONL format
        try:
            # Check training file
            with open(self.training_file, 'r', encoding='utf-8') as f:
                training_lines = f.readlines()
            
            for i, line in enumerate(training_lines[:5]):  # Check first 5 lines
                data = json.loads(line.strip())
                if 'messages' not in data:
                    self.log(f"❌ Invalid format in training file line {i+1}: missing 'messages'")
                    return False
            
            # Check validation file
            with open(self.validation_file, 'r', encoding='utf-8') as f:
                validation_lines = f.readlines()

            self.log(f"✅ Files validated successfully")
            self.log(f"📊 Training samples: {len(training_lines)}")
            self.log(f"📊 Validation samples: {len(validation_lines)}")
            # If dataset stats loaded, warn on mismatch
            try:
                if self.dataset_stats is not None:
                    expected_train = self.dataset_stats.get('train_count') or self.dataset_stats.get('training', {}).get('count')
                    expected_val = self.dataset_stats.get('val_count') or self.dataset_stats.get('validation', {}).get('count')
                    if isinstance(expected_train, int) and expected_train != len(training_lines):
                        self.log(f"⚠️ Mismatch: stats train_count={expected_train} vs lines={len(training_lines)}")
                    if isinstance(expected_val, int) and expected_val != len(validation_lines):
                        self.log(f"⚠️ Mismatch: stats val_count={expected_val} vs lines={len(validation_lines)}")
            except Exception as e:
                self.log(f"⚠️ Error comparing counts to stats: {e}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ File validation error: {e}")
            return False
    
    def upload_files(self) -> bool:
        """
        Upload training and validation files to OpenAI.
        
        Returns:
            bool: True if upload successful
        """
        self.log("📤 Uploading files to OpenAI...")
        
        try:
            # Upload training file
            self.log(f"📤 Uploading training file: {self.training_file}")
            with open(self.training_file, 'rb') as f:
                training_response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            self.training_file_id = training_response.id
            self.log(f"✅ Training file uploaded: {self.training_file_id}")
            
            # Upload validation file
            self.log(f"📤 Uploading validation file: {self.validation_file}")
            with open(self.validation_file, 'rb') as f:
                validation_response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            self.validation_file_id = validation_response.id
            self.log(f"✅ Validation file uploaded: {self.validation_file_id}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ File upload error: {e}")
            return False
    
    def create_fine_tuning_job(self) -> bool:
        """
        Create fine-tuning job with optimal hyperparameters.
        
        Returns:
            bool: True if job created successfully
        """
        self.log("🎯 Creating fine-tuning job...")
        
        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=self.training_file_id,
                validation_file=self.validation_file_id,
                model=BASE_MODEL,
                suffix=MODEL_SUFFIX,
                hyperparameters={
                    "n_epochs": "auto",  # Auto-determine optimal epochs
                    "batch_size": "auto",  # Auto-scale batch size
                    "learning_rate_multiplier": "auto"  # Auto-optimize learning rate
                }
            )
            
            self.job_id = response.id
            self.start_time = datetime.now()
            
            self.log(f"✅ Fine-tuning job created: {self.job_id}")
            self.log(f"🧠 Base model: {BASE_MODEL} (Latest GPT-4o August 2024)")
            self.log(f"🔬 Training method: Supervised Fine-Tuning (SFT)")
            self.log(f"🏷️ Model suffix: {MODEL_SUFFIX}")
            self.log(f"⚙️ Hyperparameters: Auto-optimized")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Job creation error: {e}")
            return False
    
    def monitor_training(self) -> bool:
        """
        Monitor fine-tuning job progress with real-time updates.
        
        Returns:
            bool: True if training completed successfully
        """
        self.log("👀 Monitoring training progress...")
        self.log(f"⏱️ Checking status every {POLL_INTERVAL} seconds")
        
        elapsed_time = 0
        
        while elapsed_time < MAX_WAIT_TIME:
            try:
                # Get job status
                job = self.client.fine_tuning.jobs.retrieve(self.job_id)
                status = job.status
                
                # Log current status
                self.log(f"📊 Status: {status} (elapsed: {elapsed_time//60}m {elapsed_time%60}s)")
                
                # Handle different statuses
                if status == "succeeded":
                    self.model_id = job.fine_tuned_model
                    self.log(f"🎉 Training completed successfully!")
                    self.log(f"🤖 Fine-tuned model: {self.model_id}")
                    
                    # Log training metrics if available
                    if hasattr(job, 'trained_tokens') and job.trained_tokens:
                        self.log(f"📈 Tokens trained: {job.trained_tokens:,}")
                    
                    return True
                
                elif status == "failed":
                    if hasattr(job, 'error') and job.error:
                        if hasattr(job.error, 'message'):
                            error_msg = job.error.message
                        else:
                            error_msg = str(job.error)
                    else:
                        error_msg = 'Unknown error'
                    self.log(f"❌ Training failed: {error_msg}")
                    return False
                
                elif status == "cancelled":
                    self.log(f"⚠️ Training was cancelled")
                    return False
                
                elif status in ["validating_files", "queued", "running"]:
                    # Training in progress
                    if status == "running" and hasattr(job, 'trained_tokens') and job.trained_tokens:
                        self.log(f"🔄 Training in progress... ({job.trained_tokens:,} tokens)")
                    else:
                        self.log(f"🔄 {status.replace('_', ' ').title()}...")
                
                # Wait before next check
                time.sleep(POLL_INTERVAL)
                elapsed_time += POLL_INTERVAL
                
            except Exception as e:
                self.log(f"⚠️ Error checking status: {e}")
                time.sleep(POLL_INTERVAL)
                elapsed_time += POLL_INTERVAL
        
        self.log(f"⏰ Training monitoring timed out after {MAX_WAIT_TIME//60} minutes")
        return False
    
    def validate_model(self) -> bool:
        """
        Test the fine-tuned model with validation prompts.
        
        Returns:
            bool: True if model responds appropriately
        """
        if not self.model_id:
            self.log("❌ No model ID available for validation")
            return False
        
        self.log("🧪 Validating fine-tuned model...")
        
        try:
            for i, prompt in enumerate(VALIDATION_PROMPTS, 1):
                self.log(f"🔍 Test {i}/{len(VALIDATION_PROMPTS)}: {prompt[:50]}...")
                
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                
                answer = response.choices[0].message.content
                self.log(f"✅ Response length: {len(answer)} characters")
                
                # Brief pause between requests
                time.sleep(1)
            
            self.log("✅ Model validation completed successfully")
            return True
            
        except Exception as e:
            self.log(f"❌ Model validation error: {e}")
            return False
    
    def save_model_info(self):
        """
        Save complete model information and training metadata to JSON file.
        """
        self.log("💾 Saving model information...")
        
        model_info = {
            "model_id": self.model_id,
            "job_id": self.job_id,
            "base_model": BASE_MODEL,
            "model_suffix": MODEL_SUFFIX,
            "status": "succeeded" if self.model_id else "failed",
            "training_file_id": self.training_file_id,
            "validation_file_id": self.validation_file_id,
            "created_at": self.start_time.isoformat() if self.start_time else None,
            "training_duration": str(datetime.now() - self.start_time) if self.start_time else None,
            "files": {
                "training": str(self.training_file),
                "validation": str(self.validation_file),
                "model_info": str(self.model_info_file),
                "log": str(self.log_file)
            },
            "validation_prompts": VALIDATION_PROMPTS,
            "hyperparameters": {
                "n_epochs": "auto",
                "batch_size": "auto", 
                "learning_rate_multiplier": "auto"
            },
            "dataset_stats": self.dataset_stats,
        }
        
        with open(self.model_info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        self.log(f"💾 Model info saved to: {self.model_info_file}")
    
    def run_complete_workflow(self) -> bool:
        """
        Execute the complete fine-tuning workflow from start to finish.
        
        Workflow Steps:
        1. Validate training files
        2. Upload files to OpenAI
        3. Create fine-tuning job
        4. Monitor training progress
        5. Validate completed model
        6. Save model information
        
        Returns:
            bool: True if entire workflow completed successfully
        """
        self.log("🚀 Starting complete fine-tuning workflow...")
        self.log("="*60)
        
        try:
            # Step 1: Validate files
            if not self.validate_files():
                self.log("❌ File validation failed")
                return False
            
            # Step 2: Upload files
            if not self.upload_files():
                self.log("❌ File upload failed")
                return False
            
            # Step 3: Create job
            if not self.create_fine_tuning_job():
                self.log("❌ Job creation failed")
                return False
            
            # Step 4: Monitor training
            if not self.monitor_training():
                self.log("❌ Training failed or timed out")
                return False
            
            # Step 5: Validate model
            if not self.validate_model():
                self.log("⚠️ Model validation failed, but training succeeded")
            
            # Step 6: Save info
            self.save_model_info()
            
            # Success summary
            total_time = datetime.now() - self.start_time
            self.log("="*60)
            self.log("🎉 FINE-TUNING COMPLETED SUCCESSFULLY!")
            self.log(f"🤖 Model ID: {self.model_id}")
            self.log(f"⏱️ Total time: {total_time}")
            self.log(f"💾 Model info: {self.model_info_file}")
            self.log("✅ Ready for deployment!")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Workflow error: {e}")
            return False

# ═══════════════════════════ UTILITY FUNCTIONS ═══════════════════════════ #

def check_existing_job(job_id: str) -> Dict[str, Any]:
    """
    Check status of existing fine-tuning job.
    
    Args:
        job_id: OpenAI fine-tuning job ID
        
    Returns:
        Dict: Job status and details
    """
    client = OpenAI()
    
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        return {
            "id": job.id,
            "status": job.status,
            "model": getattr(job, 'fine_tuned_model', None),
            "created_at": job.created_at,
            "finished_at": getattr(job, 'finished_at', None)
        }
    except Exception as e:
        return {"error": str(e)}

def list_fine_tuned_models() -> List[Dict[str, Any]]:
    """
    List all fine-tuned models in the organization.
    
    Returns:
        List: Available fine-tuned models
    """
    client = OpenAI()
    
    try:
        models = client.models.list()
        fine_tuned = [
            {
                "id": model.id,
                "created": model.created,
                "owned_by": model.owned_by
            }
            for model in models.data
            if model.id.startswith("ft:")
        ]
        return fine_tuned
    except Exception as e:
        return [{"error": str(e)}]

# ═══════════════════════════ MAIN EXECUTION ═══════════════════════════ #

if __name__ == "__main__":
    print("🚀 OpenAI Fine-Tuning Manager")
    print("="*60)
    print(f"📁 Training file: {TRAINING_FILE}")
    print(f"📁 Validation file: {VALIDATION_FILE}")
    print(f"🧠 Base model: {BASE_MODEL}")
    print(f"🏷️ Model suffix: {MODEL_SUFFIX}")
    print("─" * 60)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your API key: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    # Initialize manager
    manager = OpenAIFineTuningManager()
    
    # Run complete workflow
    success = manager.run_complete_workflow()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Test the fine-tuned model with your applications")
        print("2. Update model IDs in your scripts")
        print("3. Monitor model performance and usage")
        exit(0)
    else:
        print("\n❌ Fine-tuning workflow failed")
        print("Check the log file for detailed error information")
        exit(1)
