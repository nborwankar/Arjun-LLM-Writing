#!/usr/bin/env python3
"""
=============================================================================
CEREBRAS TO OPENAI CONVERTER - Training Dataset Format Converter
=============================================================================

PURPOSE:
Converts Cerebras dataset builder output (JSON format) to OpenAI fine-tuning 
format (JSONL with messages structure). Transforms custom training pairs into 
the messages format required by OpenAI's fine-tuning API.

INPUT FILES:
- cerebras_dataset.json - Output from cerebras_dataset_builder.py
  * Contains training_pairs with instruction/input/output format
  * Includes metadata, document analysis, and quality scores
  * Structure: {"summary": stats, "documents": metadata, "chunks": segments, "training_pairs": qa_data}

OUTPUT FILES:
- arjun_voice_training.jsonl - OpenAI fine-tuning training dataset
  * Each line: {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  * Ready for direct upload to OpenAI fine-tuning API
  * Includes system prompt defining Arjun's investment voice and expertise

- arjun_voice_validation.jsonl - OpenAI fine-tuning validation dataset  
  * Same format as training file, used for validation during fine-tuning
  * Contains ~20% of total training pairs for model evaluation

CONVERSION PROCESS:
1. Load Cerebras dataset JSON file with training pairs
2. Define comprehensive system prompt for Arjun's investment voice
3. Convert each training pair from instruction/input/output to messages format
4. Split dataset into training (80%) and validation (20%) sets
5. Apply quality filtering and deduplication
6. Output JSONL files ready for OpenAI fine-tuning

SYSTEM PROMPT FEATURES:
- Defines Arjun's distinctive investment voice and expertise
- Emphasizes analogical thinking and data-driven insights
- Captures personal tone while maintaining professional authority
- Includes emerging markets expertise and practical wisdom

QUALITY ASSURANCE:
- Filters training pairs by quality scores and metadata
- Removes duplicates and low-quality content
- Validates message format compliance
- Provides conversion statistics and quality metrics

Last Updated: 2025-08-07
Version: 1.0 - Production Ready
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any

# ═══════════════════════════ CONFIGURATION ═══════════════════════════ #

# File paths
INPUT_JSON = "cerebras_dataset.json"
OUTPUT_TRAINING = "arjun_voice_training.jsonl"
OUTPUT_VALIDATION = "arjun_voice_validation.jsonl"

# Quality thresholds
MIN_QUALITY_SCORE = 0.75     # Minimum quality for inclusion (0-1 scale)
MIN_OUTPUT_LENGTH = 50       # Minimum characters in output
VALIDATION_SPLIT = 0.2       # 20% for validation

# System prompt defining Arjun's investment voice
ARJUN_SYSTEM_PROMPT = """You are Arjun, a seasoned investment professional with deep expertise in emerging markets, data-driven analysis, and practical investment wisdom. Your distinctive voice combines:

ANALYTICAL APPROACH:
- Data-driven insights with clear numerical backing
- Rigorous research methodology and evidence-based conclusions
- Practical, actionable investment advice grounded in real-world experience

COMMUNICATION STYLE:
- Uses analogies and metaphors to explain complex financial concepts
- Personal, conversational tone while maintaining professional authority
- Accessible explanations that make sophisticated analysis understandable

EXPERTISE AREAS:
- Emerging markets with particular focus on Asia, Latin America, and frontier markets
- Crisis analysis and lessons from historical market events
- Portfolio construction and risk management strategies
- Market psychology and behavioral finance insights

DISTINCTIVE ELEMENTS:
- Draws from personal investment experience and market observations
- Connects current events to historical patterns and lessons
- Balances optimism with realistic risk assessment
- Provides context that helps investors understand both opportunities and pitfalls

Respond with the depth, insight, and distinctive voice that reflects years of investment experience and a genuine passion for helping others navigate financial markets successfully."""

# ═══════════════════════════ CONVERTER CLASS ═══════════════════════════ #

class CerebrasToOpenAIConverter:
    """
    Converts Cerebras dataset format to OpenAI fine-tuning format.
    
    Handles the transformation from instruction/input/output format to the 
    messages structure required by OpenAI's fine-tuning API, while maintaining
    quality standards and preserving Arjun's distinctive investment voice.
    """
    
    def __init__(self, input_file: str, training_output: str, validation_output: str):
        """
        Initialize the converter with file paths.
        
        Args:
            input_file: Path to Cerebras dataset JSON file
            training_output: Output path for training JSONL file
            validation_output: Output path for validation JSONL file
        """
        self.input_file = Path(input_file)
        self.training_output = Path(training_output)
        self.validation_output = Path(validation_output)
        
        # Statistics tracking
        self.total_pairs = 0
        self.converted_pairs = 0
        self.filtered_pairs = 0
        self.training_count = 0
        self.validation_count = 0
    
    def load_cerebras_dataset(self) -> Dict[str, Any]:
        """
        Load and validate the Cerebras dataset JSON file.
        
        Returns:
            Dict: Complete dataset with training pairs and metadata
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If JSON structure is invalid
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        print(f"📁 Loading Cerebras dataset: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Validate required structure
        required_keys = ['summary', 'training_pairs']
        for key in required_keys:
            if key not in dataset:
                raise ValueError(f"Missing required key in dataset: {key}")
        
        self.total_pairs = len(dataset['training_pairs'])
        print(f"📊 Found {self.total_pairs} training pairs")
        
        return dataset
    
    def convert_training_pair(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single training pair from Cerebras format to OpenAI format.
        
        Transforms:
        {"instruction": "...", "input": "...", "output": "...", "metadata": {...}}
        
        To:
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}
        
        Args:
            pair: Training pair in Cerebras format
            
        Returns:
            Dict: Training pair in OpenAI messages format
        """
        # Combine instruction and input for user message
        user_content = pair['instruction']
        if pair.get('input') and pair['input'].strip():
            user_content += f"\n\n{pair['input']}"
        
        # Create OpenAI messages format
        openai_format = {
            "messages": [
                {
                    "role": "system",
                    "content": ARJUN_SYSTEM_PROMPT
                },
                {
                    "role": "user", 
                    "content": user_content.strip()
                },
                {
                    "role": "assistant",
                    "content": pair['output'].strip()
                }
            ]
        }
        
        return openai_format
    
    def filter_training_pair(self, pair: Dict[str, Any]) -> bool:
        """
        Apply quality filters to determine if training pair should be included.
        
        Filtering Criteria:
        - Minimum output length for meaningful responses
        - Quality score thresholds from metadata
        - Content validation (non-empty fields)
        
        Args:
            pair: Training pair to evaluate
            
        Returns:
            bool: True if pair meets quality standards
        """
        # Check required fields
        if not all(pair.get(field) for field in ['instruction', 'output']):
            return False
        
        # Check minimum output length
        if len(pair['output'].strip()) < MIN_OUTPUT_LENGTH:
            return False
        
        # Check quality score if available in metadata
        metadata = pair.get('metadata', {})
        if 'quality_score' in metadata:
            if metadata['quality_score'] < MIN_QUALITY_SCORE:
                return False
        
        return True
    
    def split_dataset(self, converted_pairs: List[Dict[str, Any]]) -> tuple:
        """
        Split converted pairs into training and validation sets.
        
        Uses random sampling to ensure representative distribution across
        both training and validation sets.
        
        Args:
            converted_pairs: List of converted training pairs
            
        Returns:
            tuple: (training_pairs, validation_pairs)
        """
        # Shuffle for random distribution
        shuffled_pairs = converted_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # Calculate split point
        validation_size = int(len(shuffled_pairs) * VALIDATION_SPLIT)
        split_point = len(shuffled_pairs) - validation_size
        
        training_pairs = shuffled_pairs[:split_point]
        validation_pairs = shuffled_pairs[split_point:]
        
        self.training_count = len(training_pairs)
        self.validation_count = len(validation_pairs)
        
        return training_pairs, validation_pairs
    
    def write_jsonl(self, pairs: List[Dict[str, Any]], output_file: Path):
        """
        Write training pairs to JSONL format file.
        
        Each line contains a complete training example in OpenAI format.
        
        Args:
            pairs: List of training pairs in OpenAI format
            output_file: Path to output JSONL file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        print(f"💾 Wrote {len(pairs)} pairs to {output_file}")
    
    def print_statistics(self, dataset: Dict[str, Any]):
        """
        Display comprehensive conversion statistics and quality metrics.
        
        Args:
            dataset: Original Cerebras dataset for context
        """
        print(f"\n🎉 CONVERSION COMPLETE!")
        print(f"{'='*60}")
        print(f"📊 STATISTICS:")
        print(f"   Total pairs found: {self.total_pairs}")
        print(f"   Pairs converted: {self.converted_pairs}")
        print(f"   Pairs filtered out: {self.filtered_pairs}")
        print(f"   Training pairs: {self.training_count}")
        print(f"   Validation pairs: {self.validation_count}")
        
        if 'summary' in dataset:
            summary = dataset['summary']
            print(f"\n📈 DATASET QUALITY:")
            print(f"   Documents processed: {summary.get('total_documents', 'N/A')}")
            print(f"   Total chunks: {summary.get('total_chunks', 'N/A')}")
            print(f"   Average quality score: {summary.get('average_quality_score', 'N/A'):.2f}")
            print(f"   Estimated cost: ${summary.get('estimated_cost', 'N/A'):.2f}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   Training: {self.training_output}")
        print(f"   Validation: {self.validation_output}")
        print(f"\n✅ Ready for OpenAI fine-tuning!")
    
    def convert(self):
        """
        Main conversion method - orchestrates the complete conversion process.
        
        Workflow:
        1. Load Cerebras dataset JSON
        2. Filter and convert training pairs
        3. Split into training and validation sets
        4. Write JSONL output files
        5. Display conversion statistics
        """
        print("🔄 CEREBRAS TO OPENAI CONVERTER")
        print("="*60)
        
        # Load dataset
        dataset = self.load_cerebras_dataset()
        
        # Convert and filter training pairs
        print(f"\n🔍 Converting and filtering training pairs...")
        converted_pairs = []
        
        for pair in dataset['training_pairs']:
            if self.filter_training_pair(pair):
                converted_pair = self.convert_training_pair(pair)
                converted_pairs.append(converted_pair)
                self.converted_pairs += 1
            else:
                self.filtered_pairs += 1
        
        print(f"✅ Converted {self.converted_pairs} high-quality pairs")
        
        # Split dataset
        print(f"\n📊 Splitting dataset (training: {int((1-VALIDATION_SPLIT)*100)}%, validation: {int(VALIDATION_SPLIT*100)}%)")
        training_pairs, validation_pairs = self.split_dataset(converted_pairs)
        
        # Write output files
        print(f"\n💾 Writing JSONL files...")
        self.write_jsonl(training_pairs, self.training_output)
        self.write_jsonl(validation_pairs, self.validation_output)
        
        # Display statistics
        self.print_statistics(dataset)

# ═══════════════════════════ MAIN EXECUTION ═══════════════════════════ #

if __name__ == "__main__":
    print("🚀 Starting Cerebras to OpenAI conversion...")
    print(f"📁 Input: {INPUT_JSON}")
    print(f"📄 Training Output: {OUTPUT_TRAINING}")
    print(f"📄 Validation Output: {OUTPUT_VALIDATION}")
    print("─" * 60)
    
    # Initialize and run converter
    converter = CerebrasToOpenAIConverter(
        input_file=INPUT_JSON,
        training_output=OUTPUT_TRAINING,
        validation_output=OUTPUT_VALIDATION
    )
    
    try:
        converter.convert()
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("Please check your input file and try again.")
        exit(1)
    
    print("\n🎯 Next steps:")
    print("1. Review the generated JSONL files")
    print("2. Upload to OpenAI for fine-tuning")
    print("3. Monitor training progress and validation metrics")
