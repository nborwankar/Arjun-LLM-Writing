"""
=============================================================================
SCRIPT NAME: dataset_sanitizer.py
=============================================================================

INPUT FILES:
- arjun_voice_training.jsonl: Original training dataset with flagged content
- arjun_voice_validation.jsonl: Original validation dataset with flagged content

OUTPUT FILES:
- arjun_voice_training_clean.jsonl: Sanitized training dataset for OpenAI
- arjun_voice_validation_clean.jsonl: Sanitized validation dataset for OpenAI

VERSION: 1.0
LAST UPDATED: 2025-08-07
AUTHOR: Claude Code

DESCRIPTION:
Sanitizes training datasets to pass OpenAI content moderation by:
- Removing long legal disclaimers and boilerplate text
- Truncating extremely long responses (>1500 tokens)
- Removing confidential markings and internal references
- Cleaning up broken URLs and formatting issues
- Preserving Arjun's investment voice and analytical style

DEPENDENCIES:
- json
- re
- tiktoken

USAGE:
python dataset_sanitizer.py

NOTES:
- Preserves the core investment insights while removing problematic content
- Maintains conversational structure and analytical depth
- Creates OpenAI-compatible versions alongside originals
=============================================================================
"""

import json
import re
import os
import tiktoken

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough word-based estimation
        return len(text.split()) * 1.3

def sanitize_text(text):
    """
    Remove problematic content that triggers OpenAI moderation
    """
    # Remove legal disclaimers and boilerplate
    disclaimer_patterns = [
        r"should not be viewed as a current or past recommendation.*?Historic market trends are not reliable indicators of",
        r"INTERNAL ONLY:.*?distribution",
        r"Proprietary information—not for distribution.*?All rights reserved\.",
        r"For Institutional Use Only.*?Copyright.*?All rights reserved\.",
        r"This information is only as current as.*?for other reasons\.",
        r"The investment strategy and themes discussed herein.*?financial situation\.",
        r"It should not be assumed that.*?client accounts\.",
        r"There can be no assurance.*?successful\.",
    ]
    
    for pattern in disclaimer_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove broken URLs and web references
    text = re.sub(r"http://www\.barra\.com/research/BarraPub/aemt-n\.aspx.*?AM", "", text)
    text = re.sub(r"http://[^\s]+", "", text)
    
    # Remove internal references
    text = re.sub(r"INTERNAL ONLY[^\n]*", "", text)
    text = re.sub(r"Not for external distribution[^\n]*", "", text)
    
    # Clean up excessive whitespace and formatting artifacts
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Multiple newlines
    text = re.sub(r"\s+", " ", text)  # Multiple spaces
    text = re.sub(r"^\s+|\s+$", "", text)  # Leading/trailing whitespace
    
    # Remove slide references and page numbers
    text = re.sub(r"--- Slide \d+ ---", "", text)
    text = re.sub(r"\d+ of \d+ \d+/\d+/\d+ \d+:\d+ AM", "", text)
    
    return text.strip()

def truncate_response(text, max_tokens=1500):
    """
    Truncate text to stay under token limit while preserving meaning
    """
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text
    
    # Find a good breaking point (sentence end)
    sentences = text.split('. ')
    truncated = ""
    
    for sentence in sentences:
        potential = truncated + sentence + ". "
        if count_tokens(potential) > max_tokens:
            break
        truncated = potential
    
    # If we couldn't fit even one sentence, do hard truncation
    if not truncated:
        words = text.split()
        for i in range(len(words)):
            potential = " ".join(words[:i])
            if count_tokens(potential) > max_tokens:
                truncated = " ".join(words[:max(1, i-1)])
                break
    
    return truncated.strip()

def sanitize_conversation(conversation):
    """
    Sanitize a complete conversation while preserving structure
    """
    sanitized = {"messages": []}
    
    for message in conversation["messages"]:
        content = message["content"]
        
        # Sanitize content
        content = sanitize_text(content)
        
        # Truncate if too long (except system messages)
        if message["role"] != "system":
            content = truncate_response(content, max_tokens=1500)
        
        # Skip empty messages
        if content.strip():
            sanitized["messages"].append({
                "role": message["role"],
                "content": content
            })
    
    # Only return if we have at least system + user + assistant
    if len(sanitized["messages"]) >= 3:
        return sanitized
    return None

def sanitize_dataset(input_file, output_file):
    """
    Sanitize entire JSONL dataset
    """
    print(f"🧹 Sanitizing {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return 0
    
    original_count = 0
    sanitized_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            try:
                conversation = json.loads(line)
                original_count += 1
                
                sanitized = sanitize_conversation(conversation)
                if sanitized:
                    outfile.write(json.dumps(sanitized, ensure_ascii=False) + '\n')
                    sanitized_count += 1
                else:
                    print(f"⚠️ Skipped conversation {original_count} (too short after sanitization)")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON error in line {original_count}: {e}")
                continue
    
    print(f"✅ Sanitized {sanitized_count}/{original_count} conversations")
    print(f"📁 Output: {output_file}")
    return sanitized_count

def main():
    """
    Main sanitization workflow
    """
    print("🚀 STARTING OPENAI DATASET SANITIZATION")
    print("=" * 60)
    
    # File paths
    files_to_process = [
        ("arjun_voice_training.jsonl", "arjun_voice_training_clean.jsonl"),
        ("arjun_voice_validation.jsonl", "arjun_voice_validation_clean.jsonl")
    ]
    
    total_processed = 0
    
    for input_file, output_file in files_to_process:
        count = sanitize_dataset(input_file, output_file)
        total_processed += count
        print()
    
    print(f"🎯 SANITIZATION COMPLETE")
    print(f"📊 Total conversations processed: {total_processed}")
    print(f"✅ Clean datasets ready for OpenAI fine-tuning")
    print("\nNext steps:")
    print("1. Test clean datasets with OpenAI fine-tuning")
    print("2. Compare results with MLX training (which uses original data)")
    print("3. Keep both versions - clean for OpenAI, original for local training")

if __name__ == "__main__":
    main()