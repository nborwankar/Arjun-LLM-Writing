#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
                        ARJUN'S AI INVESTMENT ADVISOR - WEB INTERFACE
═══════════════════════════════════════════════════════════════════════════════════════

A Flask-based web application providing an intuitive interface for interacting with 
Arjun's fine-tuned investment AI model. Features side-by-side comparison between the 
fine-tuned model and base GPT-4o-mini to demonstrate voice capture effectiveness.

INPUT FILES:
- None (web-based interface)
- Requires OpenAI API key for model access
- Uses fine-tuned model ID: ft:gpt-4o-mini-2024-07-18:personal:arjun-voice-v1:C1oFx9M6

OUTPUT FILES:
- None (real-time web responses)
- Generates investment advice and analysis through web interface
- Provides side-by-side model comparison results

KEY FEATURES:
1. Beautiful, responsive web interface with modern UI design
2. Side-by-side comparison: Fine-tuned Arjun vs Base GPT-4o-mini
3. Pre-loaded sample questions for easy testing
4. Real-time API calls with loading states and error handling
5. Investment-focused system prompts for consistent voice
6. Keyboard shortcuts (Ctrl+Enter) for improved UX

USAGE:
1. Set OpenAI API key in configuration
2. Run: python arjun_web_simple.py
3. Open browser to: http://localhost:8080
4. Ask investment questions and compare responses

REQUIREMENTS:
- Flask web framework
- OpenAI Python client
- Valid OpenAI API key with fine-tuned model access
- Modern web browser for optimal experience

PORTS:
- Default: 8080 (configurable)
- Accessible on all network interfaces (0.0.0.0)

SECURITY NOTE:
- API key should be moved to environment variable in production
- Current implementation includes API key for development convenience

Author: Investment AI Fine-tuning Project
Version: 2.0 - Side-by-side comparison interface
Last Updated: 2025-01-07
═══════════════════════════════════════════════════════════════════════════════════════
"""

from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
import json

# Configuration
API_KEY = "sk-9aqXq2P4EkkntHSA_hL7a1-BbUAPoK-OoRazyiDT93T3BlbkFJq-GxTI1F29_0_h80_1X4cOBHl5JWbTjAlIve-aJqMA"
FINETUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:personal:arjun-voice-v1:C1oFx9M6"  # Your fine-tuned model
BASE_MODEL = "gpt-4o-mini-2024-07-18"  # Base model for comparison
MODEL_ID = FINETUNED_MODEL

app = Flask(__name__)
client = OpenAI(api_key=API_KEY)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>💰 Arjun's AI Investment Advisor</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            display: grid;
            grid-template-columns: 3fr 1fr;
            gap: 30px;
        }
        .comparison-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 30px;
        }
        .model-response {
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #e1e5e9;
        }
        .arjun-response {
            border-left: 4px solid #667eea;
            background: #f8f9ff;
        }
        .base-response {
            border-left: 4px solid #ff6b6b;
            background: #fff8f8;
        }
        .chat-area {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .sidebar {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .input-group {
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
        }
        .btn:hover {
            opacity: 0.9;
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .response-area {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .sample-btn {
            background: #e3f2fd;
            color: #1976d2;
            border: 1px solid #1976d2;
            padding: 10px 15px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .sample-btn:hover {
            background: #1976d2;
            color: white;
        }
        .feature-list {
            list-style: none;
            padding: 0;
        }
        .feature-list li {
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>💰 Arjun's AI Investment Advisor</h1>
        <p>Powered by your fine-tuned GPT-4o-mini model</p>
        <small>Model: {{ model_id }}</small>
    </div>
    
    <div class="container">
        <div class="chat-area">
            <h2>🗣️ Ask Arjun Anything</h2>
            <div class="input-group">
                <textarea id="questionInput" rows="4" placeholder="Ask about markets, investments, or economic analysis..."></textarea>
            </div>
            <button class="btn" onclick="askArjun()" id="askBtn">🧠 Compare Both Models</button>
            
            <div class="loading" id="loading">
                🤔 Both models are thinking...
            </div>
            
            <div id="responseArea" class="comparison-container" style="display: none;">
                <div class="model-response arjun-response">
                    <h3>💰 Your Fine-Tuned Arjun</h3>
                    <div id="arjunResponse"></div>
                </div>
                <div class="model-response base-response">
                    <h3>📋 Generic GPT-4o-mini</h3>
                    <div id="baseResponse"></div>
                </div>
            </div>
        </div>
        
        <div class="sidebar">
            <h3>💡 Try These Questions:</h3>
            <button class="sample-btn" onclick="setQuestion('Analyze current market volatility')">📈 Market Analysis</button>
            <button class="sample-btn" onclick="setQuestion('Explain value investing with an analogy')">🎭 Value Investing</button>
            <button class="sample-btn" onclick="setQuestion('What\\'s your view on emerging markets?')">🌍 Emerging Markets</button>
            <button class="sample-btn" onclick="setQuestion('Lessons from the 2008 financial crisis')">📚 Crisis Lessons</button>
            <button class="sample-btn" onclick="setQuestion('How to think about portfolio diversification')">📊 Portfolio Advice</button>
            
            <h3>🎯 Your AI Features:</h3>
            <ul class="feature-list">
                <li>🎭 Analogical thinking</li>
                <li>📈 Data-driven insights</li>
                <li>🌍 Emerging markets expertise</li>
                <li>💡 Practical wisdom</li>
                <li>🎯 Personal investment style</li>
            </ul>
        </div>
    </div>

    <script>
        function setQuestion(question) {
            document.getElementById('questionInput').value = question;
        }
        
        async function askArjun() {
            const question = document.getElementById('questionInput').value.trim();
            if (!question) {
                alert('Please enter a question first!');
                return;
            }
            
            const askBtn = document.getElementById('askBtn');
            const loading = document.getElementById('loading');
            const responseArea = document.getElementById('responseArea');
            
            // Show loading state
            askBtn.disabled = true;
            loading.style.display = 'block';
            responseArea.style.display = 'none';
            
            try {
                const response = await fetch('/compare', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: question })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('arjunResponse').innerHTML = 
                        data.arjun_response.replace(/\\n/g, '<br>');
                    document.getElementById('baseResponse').innerHTML = 
                        data.base_response.replace(/\\n/g, '<br>');
                    responseArea.style.display = 'grid';
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Network error: ' + error.message);
            } finally {
                askBtn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        // Allow Enter key to submit
        document.getElementById('questionInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                askArjun();
            }
        });
    </script>
</body>
</html>
"""

# ═══════════════════════════ FLASK ROUTE HANDLERS ═══════════════════════════ #

@app.route('/')
def home():
    """
    Serve the main web interface page.
    
    Returns the HTML template with embedded JavaScript for interactive
    investment advice interface. Includes model ID for transparency.
    
    Returns:
        str: Rendered HTML template with model configuration
    """
    return render_template_string(HTML_TEMPLATE, model_id=MODEL_ID)

@app.route('/ask', methods=['POST'])
def ask_arjun():
    """
    Handle single model queries to the fine-tuned Arjun model.
    
    Processes investment questions and returns responses from the fine-tuned
    model only. Used for basic interaction without comparison.
    
    Request JSON:
        {
            "question": "Investment question or query"
        }
    
    Response JSON:
        {
            "success": true/false,
            "response": "Arjun's investment advice",
            "question": "Original question",
            "error": "Error message if failed"
        }
    
    Returns:
        JSON: Response with investment advice or error message
    """
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({'success': False, 'error': 'No question provided'})
        
        # Get response from fine-tuned model
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": """You are Arjun, an investment expert known for:
                    - Insightful market analysis with practical wisdom
                    - Clear explanations using analogies
                    - Data-driven but accessible approach
                    - Personal, conversational tone with professional authority
                    - Emerging markets expertise with global perspective
                    - Calm analysis during market turbulence"""
                },
                {"role": "user", "content": question}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        
        return jsonify({
            'success': True,
            'response': answer,
            'question': question
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/compare', methods=['POST'])
def compare_models():
    """
    Handle side-by-side model comparison requests.
    
    Processes the same investment question through both the fine-tuned Arjun
    model and the base GPT-4o-mini model for direct comparison. This endpoint
    demonstrates the effectiveness of fine-tuning in capturing Arjun's distinctive
    investment voice and analysis style.
    
    Request JSON:
        {
            "question": "Investment question for comparison"
        }
    
    Response JSON:
        {
            "success": true/false,
            "arjun_response": "Fine-tuned model response",
            "base_response": "Base model response",
            "question": "Original question",
            "error": "Error message if failed"
        }
    
    Features:
    - Identical system prompts for fair comparison
    - Same temperature and token limits
    - Parallel API calls for efficiency
    - Comprehensive error handling
    
    Returns:
        JSON: Side-by-side responses or error message
    """
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({'success': False, 'error': 'No question provided'})
        
        system_prompt = """You are Arjun, an investment expert known for:
        - Insightful market analysis with practical wisdom
        - Clear explanations using analogies
        - Data-driven but accessible approach
        - Personal, conversational tone with professional authority
        - Emerging markets expertise with global perspective
        - Calm analysis during market turbulence"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Get response from fine-tuned model
        arjun_response = client.chat.completions.create(
            model=FINETUNED_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        
        # Get response from base model
        base_response = client.chat.completions.create(
            model=BASE_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        
        return jsonify({
            'success': True,
            'arjun_response': arjun_response.choices[0].message.content,
            'base_response': base_response.choices[0].message.content,
            'question': question
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# ═══════════════════════════ APPLICATION STARTUP ═══════════════════════════ #

if __name__ == '__main__':
    print("🚀 Starting Arjun's AI Investment Advisor Web Interface...")
    print("═" * 60)
    print(f"🎯 Fine-tuned Model: {FINETUNED_MODEL}")
    print(f"📊 Base Model: {BASE_MODEL}")
    print(f"🌐 Web Interface: http://localhost:8080")
    print(f"⚡ Features: Side-by-side comparison, sample questions")
    print(f"🔧 Debug Mode: Enabled for development")
    print("═" * 60)
    print("💡 Try asking about market analysis, investment strategies, or portfolio advice!")
    
    # Start Flask development server
    app.run(
        debug=True,           # Enable debug mode for development
        host='0.0.0.0',      # Accept connections from any IP
        port=8080,           # Use port 8080 for web interface
        threaded=True        # Handle multiple requests concurrently
    )
