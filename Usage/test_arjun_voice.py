#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
                           ARJUN VOICE MODEL TESTING TOOL
═══════════════════════════════════════════════════════════════════════════════════════

Quick testing tool for validating the fine-tuned Arjun investment AI model.
Provides immediate feedback on voice capture effectiveness through curated
investment scenarios designed to showcase distinctive characteristics.

INPUT FILES:
- None (uses hardcoded test prompts)
- Requires OpenAI API key for model access
- Uses specific fine-tuned model ID

OUTPUT FILES:
- None (terminal output only)
- Displays model responses to test scenarios
- Shows voice characteristics and analysis quality

TEST SCENARIOS:
1. Market Volatility Analysis
   - Tests data-driven insights and calm analytical approach
   - Evaluates practical wisdom in turbulent conditions
   
2. Asian Financial Crisis Lessons
   - Validates historical perspective and learned wisdom
   - Assesses crisis analysis and practical takeaways
   
3. Value Investing with Analogies
   - Tests analogical thinking and explanatory skills
   - Evaluates investment philosophy communication
   
4. Emerging Markets Perspective
   - Validates specialized expertise in EM investments
   - Tests distinctive regional insights and analysis

VOICE VALIDATION CHECKLIST:
✓ Uses personal, conversational tone while maintaining authority
✓ Incorporates analogies and creative explanations
✓ Shows data-driven insights with practical applications
✓ Demonstrates emerging markets expertise
✓ Exhibits calm analysis during market uncertainty
✓ Provides actionable investment wisdom
✓ Maintains distinctive reasoning patterns

USAGE:
1. Ensure OpenAI API key is valid and has model access
2. Run: python test_arjun_voice.py
3. Review responses for voice characteristics
4. Compare against expected Arjun voice patterns
5. Use for quick model validation

REQUIREMENTS:
- OpenAI Python client
- Valid OpenAI API key with fine-tuned model access
- Fine-tuned model ID configured correctly

Author: Investment AI Fine-tuning Project
Version: 2.0 - Enhanced voice validation
Last Updated: 2025-01-07
═══════════════════════════════════════════════════════════════════════════════════════
"""

from openai import OpenAI

API_KEY = "sk-9aqXq2P4EkkntHSA_hL7a1-BbUAPoK-OoRazyiDT93T3BlbkFJq-GxTI1F29_0_h80_1X4cOBHl5JWbTjAlIve-aJqMA"
MODEL_ID = "ft:gpt-4o-mini-2024-07-18:personal:arjun-voice-v1:C1oFx9M6"

# ═══════════════════════════ TESTING FUNCTIONS ═══════════════════════════ #

def test_arjun_voice():
    """
    Execute comprehensive testing of the fine-tuned Arjun voice model.
    
    Runs through curated investment scenarios designed to validate voice
    capture effectiveness. Each test prompt targets specific aspects of
    Arjun's distinctive investment analysis style and expertise.
    
    Test Coverage:
    - Market volatility analysis with calm, data-driven approach
    - Historical crisis lessons demonstrating learned wisdom
    - Value investing explanations using analogical thinking
    - Emerging markets expertise and regional insights
    
    Features:
    - Professional system prompt matching training data
    - Appropriate response length and creativity settings
    - Clear formatting for easy evaluation
    - Error handling for API failures
    - Sequential testing with clear separation
    
    Voice Validation:
    Responses should demonstrate Arjun's characteristic:
    - Personal yet professional tone
    - Analogical explanations of complex concepts
    - Data-driven insights with practical applications
    - Emerging markets specialization
    - Calm analysis during uncertainty
    - Actionable investment wisdom
    """
    client = OpenAI(api_key=API_KEY)
    
    test_prompts = [
        "Analyze the current market volatility and provide your investment perspective.",
        "What lessons can we learn from the Asian financial crisis?",
        "Explain the concept of value investing using an analogy.",
        "What's your view on emerging markets in the current environment?"
    ]
    
    print("🧪 TESTING YOUR FINE-TUNED ARJUN VOICE MODEL")
    print("=" * 60)
    print(f"🎯 Model: {MODEL_ID}")
    print()
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"{i}. Testing prompt: {prompt}")
        print("-" * 40)
        
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are Arjun, an investment expert known for insightful market analysis, practical wisdom, and clear explanations using analogies."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            print(response.choices[0].message.content)
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            
        print("=" * 60)
        print()

# ═══════════════════════════ MAIN EXECUTION ═══════════════════════════ #

if __name__ == "__main__":
    print("🚀 Starting Arjun Voice Model Testing...")
    print("═" * 60)
    print(f"🎯 Model ID: {MODEL_ID}")
    print("🔍 Purpose: Validate voice capture and response quality")
    print("═" * 60)
    
    test_arjun_voice()
    
    print("═" * 60)
    print("🎉 Testing Complete!")
    print("💡 Review responses for voice characteristics and quality")
    print("🔄 Use compare_models.py for detailed side-by-side analysis")
    print("═" * 60)
