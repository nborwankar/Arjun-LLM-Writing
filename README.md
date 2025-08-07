# 💰 Arjun's AI Investment Voice - Fine-Tuned Model

A personalized GPT-4o-mini model fine-tuned on Arjun's investment writing to capture his distinctive voice, reasoning style, and market insights.

## 🎯 Model Information
- **Model ID**: `ft:gpt-4o-mini-2024-07-18:personal:arjun-voice-v1:C1oFx9M6`
- **Base Model**: GPT-4o-mini-2024-07-18
- **Training Data**: 71 samples from 28 investment documents
- **Fine-tuning Cost**: ~$0.22
- **Training Time**: ~15 minutes
- **Validation Loss**: Converged successfully

## 📚 Training Process Overview

### **1. Document Collection & Analysis**
- **Source**: 28 investment documents (PDFs, DOCX, PPTX)
- **Content**: Market commentary, research papers, strategy documents, crisis analysis
- **Total Words**: ~46,000 words across diverse document types
- **Quality**: Mix of small (1-2 pages), medium (5-10 pages), and large (15+ pages) documents

### **2. Dataset Generation Pipeline**
- **Primary Tool**: Cerebras dataset builder (`cerebras_dataset_builder.py`)
- **Alternative**: Claude Opus builder (`claude_opus_dataset_builder_v2.py`) for premium quality
- **Process**: 
  - Document extraction with PyMuPDF (PDFs) and python-docx (Word docs)
  - Intelligent chunking (500-1500 words) preserving reasoning flow
  - Multiple training pair generation per chunk (style replication, continuation, analogies)
  - Quality scoring and deduplication
  - Rich metadata capture (voice markers, document balance)

### **3. Training Data Preparation**
- **Format**: OpenAI JSONL format with instruction-input-output structure
- **Training Set**: 56 samples (`arjun_voice_training.jsonl`)
- **Validation Set**: 15 samples (`arjun_voice_validation.jsonl`)
- **Split**: 80/20 train/validation ratio
- **Token Count**: ~185K training tokens, ~58K validation tokens

### **4. Fine-Tuning Configuration**
- **Platform**: OpenAI Fine-Tuning API
- **Epochs**: 3 (optimal for dataset size)
- **Batch Size**: Auto-selected by OpenAI
- **Learning Rate**: Auto-optimized
- **Monitoring**: Real-time loss tracking and validation metrics

## 🛠️ Quick Setup

### **Prerequisites**
```bash
# Install required packages
pip install openai flask jupyter pandas matplotlib

# Set environment variable (optional)
export OPENAI_API_KEY="your-api-key-here"
```

### **Verify Installation**
```bash
# Test the model
python test_arjun_voice.py
```

## 🚀 Available Applications

### 1. **🌐 Web Interface** (Recommended)
```bash
python arjun_web_simple.py
```
**Features:**
- Opens at http://localhost:8080
- **Side-by-side comparison** with base GPT-4o-mini
- Beautiful responsive UI with gradient design
- Pre-loaded sample questions
- Real-time response streaming
- Clear visual distinction between your voice and generic AI

**Perfect for:** Demonstrating voice differences, client presentations, interactive exploration

### 2. **💻 Command Line Interface**
```bash
# Quick question
python arjun_cli.py "What's your take on emerging markets?"

# Interactive mode
python arjun_cli.py --interactive

# Custom parameters
python arjun_cli.py --temperature 0.8 --max-tokens 500 "Your question"
```
**Features:**
- Fast terminal-based interaction
- Batch processing capabilities
- Scriptable for automation
- Custom temperature and token controls
- Interactive chat mode

**Perfect for:** Quick queries, scripting, automation, power users

### 3. **📓 Jupyter Notebook Integration**
```bash
# Launch demo notebook
jupyter notebook Arjun_AI_Demo.ipynb

# Or use the helper class
from arjun_notebook_helper import ArjunAI
arjun = ArjunAI()
arjun.ask("Analyze this market data...")
```
**Features:**
- Interactive analysis environment
- Data visualization integration
- Conversation history tracking
- Export capabilities
- Rich markdown formatting
- Integration with pandas/matplotlib

**Perfect for:** Research, data analysis, report generation, exploratory analysis

### 4. **🔬 Direct API Testing**
```python
# Simple test
python test_arjun_voice.py

# Or import and use
from test_arjun_voice import test_arjun_voice
test_arjun_voice()
```
**Features:**
- Direct OpenAI API calls
- Pre-configured test questions
- Response timing and token usage
- Error handling examples

**Perfect for:** Development, debugging, integration testing

### 5. **⚖️ Model Comparison Tool**
```bash
python compare_models.py
```
**Features:**
- Batch comparison with base model
- Side-by-side text output
- Multiple test questions
- Performance metrics
- Difference highlighting

**Perfect for:** Evaluation, quality assessment, voice validation

## 📊 Performance Metrics

### **Voice Capture Success**
- **Analogical Thinking**: Successfully captures creative comparisons
- **Personal Tone**: "I think...", "More importantly..." patterns preserved
- **Market Insights**: Specific, opinionated takes vs generic advice
- **Practical Wisdom**: Actionable advice over theoretical concepts
- **Emerging Markets**: Specialized knowledge and perspective
- **Crisis Experience**: Historical lessons and calm analysis

### **Comparison Results**
**Your Fine-Tuned Model:**
- Concise, confident responses
- Personal opinions and strong takes
- Specific market references
- Analogical explanations
- Conversational yet authoritative

**Base GPT-4o-mini:**
- Generic, cautious language
- Academic, textbook-style responses
- Numbered lists and formal structure
- Longer, more verbose explanations
- No personal perspective

## 📁 File Structure

### **Core Usage Files**
- `arjun_web_simple.py` - Flask web app with side-by-side comparison
- `arjun_cli.py` - Command-line interface
- `test_arjun_voice.py` - Simple API testing script
- `compare_models.py` - Batch comparison tool
- `arjun_notebook_helper.py` - Jupyter integration utilities
- `Arjun_AI_Demo.ipynb` - Demo notebook with examples

### **Training Data & Metadata**
- `arjun_voice_training.jsonl` - Final training dataset (56 samples)
- `arjun_voice_validation.jsonl` - Validation dataset (15 samples)
- `arjun_voice_model_info.json` - Model metadata and job info
- `Writing/` - Source documents (28 PDFs, DOCX, PPTX files)

### **Dataset Generation**
- `cerebras_dataset_builder.py` - Fast, cost-effective dataset builder
- `claude_opus_dataset_builder_v2.py` - Premium quality dataset builder

### **Documentation**
- `PRD.MD` - Original project requirements
- `jupyter_setup.md` - Jupyter environment setup guide
- `requirements.txt` - Python dependencies

## 🎭 Voice Characteristics

Your fine-tuned model captures:
- **Analogical thinking** - Creative, memorable comparisons
- **Personal tone** - Conversational yet authoritative
- **Data-driven insights** - References to studies and research
- **Practical wisdom** - Actionable investment advice
- **Emerging markets expertise** - Global perspective with EM focus
- **Crisis experience** - Historical lessons and calm analysis

## 💡 Sample Questions

Try these to see your distinctive voice:
- "What's your view on emerging markets right now?"
- "Explain value investing with an analogy"
- "How should I think about market volatility?"
- "Lessons from the 2008 financial crisis"
- "Analyze the current dollar strength impact"

## 🔧 Dependencies

```bash
# Core requirements
pip install openai>=1.0.0
pip install flask>=2.0.0
pip install jupyter>=1.0.0

# Optional for data analysis
pip install pandas>=1.5.0
pip install matplotlib>=3.5.0

# Dataset generation (if regenerating)
pip install PyMuPDF>=1.23.0  # PDF extraction
pip install python-docx>=0.8.11  # Word docs
pip install python-pptx>=0.6.21  # PowerPoint
pip install cerebras-cloud-sdk>=1.0.0  # Cerebras API
pip install anthropic>=0.7.0  # Claude API
```

### **Environment Setup**
```bash
# Set API keys (optional, can be hardcoded)
export OPENAI_API_KEY="sk-your-key-here"
export CEREBRAS_API_KEY="your-cerebras-key"  # If regenerating
export ANTHROPIC_API_KEY="your-claude-key"  # If regenerating
```

## 🔄 Advanced Usage

### **Regenerating the Dataset**
If you want to add more documents or improve the training data:

```bash
# Using Cerebras (fast, cost-effective)
python cerebras_dataset_builder.py

# Using Claude Opus (premium quality)
python claude_opus_dataset_builder_v2.py
```

**Process:**
1. Add new documents to the `Writing/` folder
2. Run your preferred dataset builder
3. Convert to OpenAI format if needed
4. Upload and fine-tune a new model

### **Customizing the Voice**
To modify the AI's personality or focus:

1. **Edit System Prompts**: Update the system messages in each application
2. **Adjust Temperature**: Higher values (0.8-1.0) for more creativity
3. **Modify Training Data**: Add specific examples of desired behavior
4. **Fine-tune Parameters**: Experiment with epochs and learning rates

### **Integration Examples**
```python
# Custom integration
from openai import OpenAI

client = OpenAI(api_key="your-key")
response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:personal:arjun-voice-v1:C1oFx9M6",
    messages=[
        {"role": "system", "content": "You are Arjun, an investment expert..."},
        {"role": "user", "content": "Your question here"}
    ],
    max_tokens=800,
    temperature=0.7
)
```

## 🎉 Success Metrics

The fine-tuning successfully captured your voice as evidenced by:
- **✅ Clear differences** in side-by-side comparisons
- **✅ Personal, opinionated responses** vs generic advice
- **✅ Analogical thinking** and conversational tone
- **✅ Specific market insights** and confident assertions
- **✅ Practical, actionable** investment wisdom
- **✅ Emerging markets expertise** and global perspective
- **✅ Crisis experience** and historical lessons

## 💰 Cost Breakdown
- **Dataset Generation**: $15.60 (Cerebras) or $173 (Claude Opus)
- **Fine-Tuning**: $0.22 (OpenAI)
- **Total Investment**: ~$16-173 for your personalized AI advisor

**Your AI investment advisor is ready to provide insights in your distinctive voice!** 🚀📈
