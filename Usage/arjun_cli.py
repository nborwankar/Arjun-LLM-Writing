#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
                    ARJUN'S AI INVESTMENT ADVISOR - COMMAND LINE INTERFACE
═══════════════════════════════════════════════════════════════════════════════════════

A command-line interface for interacting with Arjun's fine-tuned investment AI model.
Provides both single-query and interactive conversation modes for terminal-based access
to investment analysis and advice.

INPUT FILES:
- None (command-line arguments and user input)
- Requires OpenAI API key for model access
- Uses fine-tuned model: ft:gpt-4o-mini-2024-07-18:personal:arjun-voice-v1:C1oFx9M6

OUTPUT FILES:
- None (terminal output only)
- Displays investment advice and analysis in terminal
- Maintains conversation history during interactive sessions

USAGE MODES:
1. Single Query Mode:
   python arjun_cli.py "What's your view on current market volatility?"
   
2. Interactive Mode:
   python arjun_cli.py --interactive
   
3. Custom Parameters:
   python arjun_cli.py --max-tokens 800 --temperature 0.5 "Explain value investing"

COMMAND LINE OPTIONS:
- prompt: Investment question (positional argument)
- --interactive, -i: Start interactive conversation mode
- --max-tokens: Maximum response length (default: 500)
- --temperature: Response creativity 0.0-1.0 (default: 0.7)

INTERACTIVE COMMANDS:
- quit/exit/q: Exit the program
- help: Show available commands and sample questions
- history: Display conversation history
- clear: Clear conversation history

FEATURES:
1. Single-shot queries for quick investment questions
2. Interactive mode with conversation history and context
3. Customizable response parameters (length, creativity)
4. Built-in help system with sample questions
5. Conversation history management
6. Error handling and graceful exits
7. Investment-focused system prompts

REQUIREMENTS:
- Python 3.7+
- OpenAI Python client
- Valid OpenAI API key with fine-tuned model access
- Terminal/command prompt access

SECURITY NOTE:
- API key should be moved to environment variable in production
- Current implementation includes API key for development convenience

Author: Investment AI Fine-tuning Project
Version: 2.0 - Interactive mode with history
Last Updated: 2025-01-07
═══════════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import sys
from openai import OpenAI

API_KEY = "sk-9aqXq2P4EkkntHSA_hL7a1-BbUAPoK-OoRazyiDT93T3BlbkFJq-GxTI1F29_0_h80_1X4cOBHl5JWbTjAlIve-aJqMA"
MODEL_ID = "ft:gpt-4o-mini-2024-07-18:personal:arjun-voice-v1:C1oFx9M6"  # Your fine-tuned model

# ═══════════════════════════ MAIN APPLICATION LOGIC ═══════════════════════════ #

def main():
    """
    Main entry point for the CLI application.
    
    Parses command line arguments and routes to appropriate mode:
    - Single query mode for quick questions
    - Interactive mode for extended conversations
    - Help display for usage information
    
    Command Line Arguments:
        prompt: Investment question (positional)
        --interactive: Enable interactive conversation mode
        --max-tokens: Maximum response tokens (default: 500)
        --temperature: Response creativity 0.0-1.0 (default: 0.7)
    """
    parser = argparse.ArgumentParser(description="Arjun's AI Investment Advisor CLI")
    parser.add_argument("prompt", nargs="*", help="Your investment question")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max response tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Response creativity")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args)
    elif args.prompt:
        prompt = " ".join(args.prompt)
        get_response(prompt, args)
    else:
        print("💰 Arjun's AI Investment Advisor")
        print("Usage: python arjun_cli.py 'your question'")
        print("   or: python arjun_cli.py --interactive")
        sys.exit(1)

def get_response(prompt, args):
    """
    Get a single response from Arjun's fine-tuned investment AI model.
    
    Processes a single investment question and returns Arjun's analysis
    without maintaining conversation history. Ideal for quick queries.
    
    Args:
        prompt (str): Investment question or query
        args: Parsed command line arguments with max_tokens and temperature
        
    Features:
    - Investment-focused system prompt
    - Customizable response parameters
    - Error handling with user-friendly messages
    - Professional formatting of responses
    """
    try:
        client = OpenAI(api_key=API_KEY)
        
        print(f"🤔 Arjun is analyzing your question...")
        
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "You are Arjun, an investment expert known for insightful market analysis, practical wisdom, and clear explanations using analogies."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        print(f"\n💰 Arjun's Response:")
        print("=" * 50)
        print(response.choices[0].message.content)
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def interactive_mode(args):
    """
    Start interactive conversation mode with Arjun's AI.
    
    Provides a continuous conversation interface with:
    - Conversation history for context
    - Built-in commands (help, history, clear, quit)
    - Context-aware responses using recent exchanges
    - Graceful error handling and recovery
    
    Args:
        args: Parsed command line arguments with response parameters
        
    Interactive Commands:
    - quit/exit/q: Exit the program
    - help: Show available commands and sample questions  
    - history: Display conversation history
    - clear: Clear conversation history
    
    Features:
    - Maintains last 4 exchanges for context
    - Professional conversation flow
    - Investment-focused interactions
    - Keyboard interrupt handling
    """
    print("💰 Arjun's AI Investment Advisor - Interactive Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("=" * 50)
    
    client = OpenAI(api_key=API_KEY)
    history = []
    
    while True:
        try:
            prompt = input("\n🗣️  You: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using Arjun's AI!")
                break
                
            elif prompt.lower() == 'help':
                show_help()
                continue
                
            elif prompt.lower() == 'history':
                show_history(history)
                continue
                
            elif prompt.lower() == 'clear':
                history.clear()
                print("🗑️  History cleared")
                continue
                
            elif not prompt:
                continue
            
            print("🤔 Arjun is thinking...")
            
            # Include recent history for context
            messages = [
                {
                    "role": "system",
                    "content": "You are Arjun, an investment expert known for insightful market analysis, practical wisdom, and clear explanations using analogies."
                }
            ]
            
            # Add recent conversation history (last 4 exchanges)
            for exchange in history[-4:]:
                messages.append({"role": "user", "content": exchange["user"]})
                messages.append({"role": "assistant", "content": exchange["assistant"]})
            
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            answer = response.choices[0].message.content
            
            print(f"\n💰 Arjun: {answer}")
            
            # Save to history
            history.append({
                "user": prompt,
                "assistant": answer
            })
            
        except KeyboardInterrupt:
            print("\n👋 Thanks for using Arjun's AI!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def show_help():
    """
    Display help information for interactive mode.
    
    Shows available commands and provides sample investment questions
    to help users get started with meaningful interactions.
    """
    print("""
💡 Available commands:
   quit/exit/q  - Exit the program
   help         - Show this help
   history      - Show conversation history  
   clear        - Clear conversation history
   
💰 Sample questions:
   "What's your view on current market volatility?"
   "Explain value investing using an analogy"
   "How should I think about emerging markets?"
   "What lessons from the 2008 crisis still apply?"
    """)

def show_history(history):
    """
    Display the conversation history in a formatted manner.
    
    Args:
        history (list): List of conversation exchanges with user/assistant pairs
        
    Shows numbered exchanges with truncated responses for easy scanning.
    Handles empty history gracefully.
    """
    if not history:
        print("📝 No conversation history yet")
        return
    
    print("\n📚 Conversation History:")
    print("=" * 50)
    for i, exchange in enumerate(history, 1):
        print(f"\n{i}. You: {exchange['user']}")
        print(f"   Arjun: {exchange['assistant'][:100]}...")
    print("=" * 50)

# ═══════════════════════════ APPLICATION ENTRY POINT ═══════════════════════════ #

if __name__ == "__main__":
    main()
