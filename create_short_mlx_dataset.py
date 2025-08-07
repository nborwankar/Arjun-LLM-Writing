#!/usr/bin/env python3
"""
Create shorter MLX training dataset by truncating long sequences
"""

import json
import os

def truncate_text(text, max_tokens=512):
    """Truncate text to approximately max_tokens"""
    words = text.split()
    # Rough approximation: 1 token ≈ 0.75 words
    max_words = int(max_tokens * 0.75)
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + "..."

def create_short_dataset():
    """Create shorter training sequences"""
    
    # Read original training data
    with open('arjun_voice_training.jsonl', 'r') as f:
        training_data = [json.loads(line) for line in f]
    
    with open('arjun_voice_validation.jsonl', 'r') as f:
        validation_data = [json.loads(line) for line in f]
    
    # Create short MLX dataset directory
    os.makedirs('mlx_short_dataset', exist_ok=True)
    
    # Process training data
    short_train = []
    for item in training_data:
        messages = item['messages']
        
        # Extract system, user, and assistant messages
        system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
        assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
        
        # Create shorter system prompt
        short_system = "You are Arjun, an experienced investment professional specializing in emerging markets. Provide data-driven insights with analogies and practical advice."
        
        # Truncate assistant response to manageable length
        short_assistant = truncate_text(assistant_msg, max_tokens=400)
        
        # Create MLX format text
        text = f"System: {short_system}\n\nHuman: {user_msg}\n\nAssistant: {short_assistant}"
        
        short_train.append({"text": text})
    
    # Process validation data similarly
    short_val = []
    for item in validation_data:
        messages = item['messages']
        
        system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
        assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
        
        short_system = "You are Arjun, an experienced investment professional specializing in emerging markets. Provide data-driven insights with analogies and practical advice."
        short_assistant = truncate_text(assistant_msg, max_tokens=400)
        
        text = f"System: {short_system}\n\nHuman: {user_msg}\n\nAssistant: {short_assistant}"
        
        short_val.append({"text": text})
    
    # Write short datasets
    with open('mlx_short_dataset/train.jsonl', 'w') as f:
        for item in short_train:
            f.write(json.dumps(item) + '\n')
    
    with open('mlx_short_dataset/valid.jsonl', 'w') as f:
        for item in short_val:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Created short MLX dataset:")
    print(f"   Training: {len(short_train)} samples")
    print(f"   Validation: {len(short_val)} samples")
    print(f"   Max sequence length: ~600 tokens")

if __name__ == "__main__":
    create_short_dataset()
