#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════
                        ARJUN AI JUPYTER NOTEBOOK HELPER
═══════════════════════════════════════════════════════════════════════════════════════

Comprehensive Jupyter notebook integration for Arjun's fine-tuned investment AI model.
Provides rich, interactive interface for data analysis, market commentary, and investment
research workflows with beautiful formatting and conversation management.

INPUT FILES:
- None (interactive notebook usage)
- Requires OpenAI API key for model access
- Uses fine-tuned Arjun voice model ID
- Accepts pandas DataFrames for data analysis

OUTPUT FILES:
- arjun_conversation.txt (optional export of conversation history)
- Generated visualizations (matplotlib/seaborn charts)
- Rich HTML formatted responses in notebook cells

CORE FEATURES:

1. INTERACTIVE Q&A:
   - ask(): General investment questions with formatted responses
   - Conversation history tracking and management
   - Markdown rendering for beautiful notebook display

2. DATA ANALYSIS INTEGRATION:
   - analyze_data(): DataFrame analysis with statistical summaries
   - data_story(): Automated visualization with narrative interpretation
   - Seamless pandas integration for financial datasets

3. SPECIALIZED INVESTMENT FUNCTIONS:
   - market_commentary(): Current market analysis and perspectives
   - explain_with_analogy(): Complex concepts with memorable analogies
   - crisis_lessons(): Historical market event insights
   - portfolio_advice(): Portfolio construction guidance
   - emerging_markets_insight(): EM expertise and regional analysis
   - compare_scenarios(): Side-by-side investment scenario evaluation

4. CONVERSATION MANAGEMENT:
   - show_conversation_history(): Formatted history display
   - clear_history(): Reset conversation state
   - export_conversation(): Save discussions to text file

5. VISUALIZATION CAPABILITIES:
   - Automatic chart generation for numeric data
   - Correlation heatmaps for multi-variable analysis
   - Distribution plots and statistical visualizations
   - Integration with matplotlib and seaborn

ARJUN VOICE CHARACTERISTICS:
✓ Data-driven insights with practical applications
✓ Analogical thinking for complex concept explanations
✓ Emerging markets expertise and regional insights
✓ Personal conversational tone with professional authority
✓ Historical perspective and crisis wisdom
✓ Calm analysis during market uncertainty
✓ Actionable investment recommendations

USAGE EXAMPLES:

# Basic setup
arjun = ArjunAI()

# Ask investment questions
arjun.ask("What's your view on current market volatility?")

# Analyze financial data
df = pd.read_csv('stock_data.csv')
arjun.analyze_data(df, "What trends do you see in this data?")

# Get market commentary
arjun.market_commentary("emerging markets in 2024")

# Create data stories with visualizations
arjun.data_story(df, "Stock performance over time")

# Compare investment scenarios
arjun.compare_scenarios("Growth stocks", "Value stocks")

REQUIREMENTS:
- Jupyter notebook environment
- OpenAI Python client
- pandas for data manipulation
- matplotlib and seaborn for visualization
- IPython for rich display capabilities
- Valid OpenAI API key with fine-tuned model access

SECURITY NOTES:
- API key currently hardcoded for development convenience
- Recommend moving to environment variables for production
- Conversation history stored in memory only (not persistent)

Author: Investment AI Fine-tuning Project
Version: 2.0 - Enhanced notebook integration
Last Updated: 2025-01-07
═══════════════════════════════════════════════════════════════════════════════════════
"""

from openai import OpenAI
import pandas as pd
from IPython.display import display, Markdown, HTML
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')  # Suppress routine warnings for cleaner notebook output

# ═══════════════════════════ MAIN ARJUN AI CLASS ═══════════════════════════ #

class ArjunAI:
    """
    Arjun's AI Investment Advisor for Jupyter notebooks.
    
    A comprehensive wrapper class that provides rich, interactive access to the
    fine-tuned Arjun investment AI model within Jupyter notebook environments.
    Combines conversational AI with data analysis capabilities, visualization
    tools, and specialized investment functions.
    
    Key Features:
    - Interactive Q&A with beautiful markdown formatting
    - DataFrame analysis with statistical summaries
    - Automated visualization with narrative interpretation
    - Specialized investment analysis functions
    - Conversation history management
    - Export capabilities for research documentation
    
    Attributes:
        client (OpenAI): OpenAI API client instance
        model_id (str): Fine-tuned Arjun voice model identifier
        conversation_history (list): Stored Q&A exchanges
    
    Usage:
        arjun = ArjunAI()
        arjun.ask("What's your market outlook?")
        arjun.analyze_data(df, "What patterns do you see?")
    """
    
    def __init__(self, api_key="sk-9aqXq2P4EkkntHSA_hL7a1-BbUAPoK-OoRazyiDT93T3BlbkFJq-GxTI1F29_0_h80_1X4cOBHl5JWbTjAlIve-aJqMA"):
        """
        Initialize Arjun AI instance with API configuration and welcome display.
        
        Sets up OpenAI client, model configuration, and conversation tracking.
        Displays a formatted welcome message in the notebook with feature overview
        and usage instructions.
        
        Args:
            api_key (str): OpenAI API key for model access
            
        Features Initialized:
        - OpenAI client with API authentication
        - Fine-tuned model ID configuration
        - Empty conversation history list
        - Rich HTML welcome message display
        """
        self.client = OpenAI(api_key=api_key)
        self.model_id = "ft:gpt-4o-mini-2024-07-18:personal:arjun-voice-v1:C1oFx9M6"  # Your fine-tuned model
        self.conversation_history = []
        
        # Display welcome message
        display(HTML("""
        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; background-color: #f9f9f9;">
        <h3>💰 Arjun's AI Investment Advisor - Jupyter Edition</h3>
        <p><strong>Your fine-tuned investment expert is ready!</strong></p>
        <p>🎯 <strong>Features:</strong> Market analysis, analogies, emerging markets insights</p>
        <p>💡 <strong>Try:</strong> arjun.ask("your question"), arjun.market_commentary("topic")</p>
        </div>
        """))
        
    def ask(self, question, max_tokens=600, temperature=0.7, show_markdown=True):
        """
        Ask Arjun a question and receive a formatted response.
        
        Core interaction method that sends questions to the fine-tuned model
        and displays responses with beautiful markdown formatting. Automatically
        tracks conversation history for reference and export.
        
        Args:
            question (str): Investment question or topic for analysis
            max_tokens (int): Maximum response length (default: 600)
            temperature (float): Response creativity level (default: 0.7)
            show_markdown (bool): Display formatted output (default: True)
            
        Returns:
            str or None: Response text if show_markdown=False, None otherwise
            
        Features:
        - Professional system prompt matching training data
        - Automatic conversation history tracking
        - Rich markdown display with user/AI formatting
        - Error handling with user-friendly messages
        - Flexible output options for programmatic use
        
        Example:
            arjun.ask("What's your view on emerging markets?")
            response = arjun.ask("Analyze this trend", show_markdown=False)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": """You are Arjun, an investment expert known for:
                        - Insightful market analysis with data-driven insights
                        - Clear explanations using analogies
                        - Practical, actionable investment advice
                        - Emerging markets expertise
                        - Calm, professional tone with personal touch"""
                    },
                    {"role": "user", "content": question}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            
            # Save to history
            self.conversation_history.append({
                'question': question,
                'answer': answer
            })
            
            if show_markdown:
                display(Markdown(f"**🗣️ You:** {question}"))
                display(Markdown(f"**💰 Arjun:** {answer}"))
            else:
                return answer
                
        except Exception as e:
            display(HTML(f'<div style="color: red;">❌ Error: {e}</div>'))
            return None
    
    def analyze_data(self, df, question_about_data):
        """
        Analyze a pandas DataFrame and get Arjun's investment insights.
        
        Combines statistical analysis with AI interpretation to provide
        meaningful investment insights from financial data. Automatically
        generates data summaries and statistical descriptions.
        
        Args:
            df (pd.DataFrame): Financial dataset to analyze
            question_about_data (str): Specific question about the data
            
        Returns:
            Formatted response with data context and investment insights
            
        Features:
        - Automatic data shape and column analysis
        - Sample data preview for context
        - Statistical summary generation
        - Investment-focused interpretation
        
        Example:
            df = pd.read_csv('stock_prices.csv')
            arjun.analyze_data(df, "What trends indicate buying opportunities?")
        """
        # Create data summary
        summary = f"""
        Dataset Summary:
        - Shape: {df.shape[0]} rows, {df.shape[1]} columns
        - Columns: {list(df.columns)}
        - Sample data:
        {df.head(3).to_string()}
        
        Basic statistics:
        {df.describe().to_string()}
        """
        
        full_question = f"{question_about_data}\n\nHere's the data context:\n{summary}"
        return self.ask(full_question)
    
    def market_commentary(self, topic):
        """
        Get Arjun's professional market commentary on specific topics.
        
        Specialized function for market analysis and investment perspectives.
        Optimized prompt structure for comprehensive market insights.
        
        Args:
            topic (str): Market topic, sector, or economic theme
            
        Returns:
            Formatted market analysis with investment implications
            
        Example:
            arjun.market_commentary("Federal Reserve policy changes")
            arjun.market_commentary("emerging markets volatility")
        """
        prompt = f"Provide your investment perspective and market commentary on: {topic}"
        return self.ask(prompt)
    
    def explain_with_analogy(self, concept):
        """
        Get clear explanations of complex financial concepts using analogies.
        
        Leverages Arjun's distinctive analogical thinking to make complex
        investment concepts accessible and memorable.
        
        Args:
            concept (str): Financial concept or investment term to explain
            
        Returns:
            Analogy-based explanation with practical applications
            
        Example:
            arjun.explain_with_analogy("portfolio diversification")
            arjun.explain_with_analogy("compound interest")
        """
        prompt = f"Explain {concept} using a clear, memorable analogy that makes it easy to understand."
        return self.ask(prompt)
    
    def crisis_lessons(self, crisis_or_event):
        """
        Extract investment lessons from historical market events.
        
        Taps into Arjun's historical perspective and crisis wisdom
        to provide actionable insights from past market events.
        
        Args:
            crisis_or_event (str): Historical market event or crisis
            
        Returns:
            Historical analysis with current investment applications
            
        Example:
            arjun.crisis_lessons("2008 financial crisis")
            arjun.crisis_lessons("Asian financial crisis")
        """
        prompt = f"What key lessons can investors learn from {crisis_or_event}? Focus on practical, actionable insights."
        return self.ask(prompt)
    
    def portfolio_advice(self, portfolio_context):
        """
        Get personalized portfolio construction and management advice.
        
        Provides strategic guidance on portfolio allocation, risk management,
        and investment selection based on specific context and goals.
        
        Args:
            portfolio_context (str): Portfolio situation, goals, or constraints
            
        Returns:
            Strategic portfolio guidance with actionable recommendations
            
        Example:
            arjun.portfolio_advice("Conservative retiree seeking income")
            arjun.portfolio_advice("Young investor with high risk tolerance")
        """
        prompt = f"Given this portfolio context: {portfolio_context}, what's your advice on construction, diversification, and risk management?"
        return self.ask(prompt)
    
    def emerging_markets_insight(self, specific_market=None):
        """
        Access Arjun's specialized emerging markets expertise.
        
        Leverages deep EM knowledge for regional analysis, opportunities,
        and risk assessment in developing market investments.
        
        Args:
            specific_market (str, optional): Specific country or region focus
            
        Returns:
            EM analysis with regional insights and investment implications
            
        Example:
            arjun.emerging_markets_insight()  # General EM outlook
            arjun.emerging_markets_insight("Southeast Asia")
            arjun.emerging_markets_insight("Latin American markets")
        """
        if specific_market:
            prompt = f"What's your current perspective on {specific_market} as an emerging market investment opportunity?"
        else:
            prompt = "What's your current view on emerging markets as an asset class? Key opportunities and risks?"
        return self.ask(prompt)
    
    def data_story(self, df, chart_description=""):
        """
        Create compelling data narratives with automated visualizations.
        
        Combines statistical analysis, chart generation, and AI narrative
        to tell the investment story hidden in your data. Automatically
        selects appropriate visualizations based on data characteristics.
        
        Args:
            df (pd.DataFrame): Dataset for analysis and visualization
            chart_description (str): Optional context about the expected chart
            
        Returns:
            Formatted response with data insights and investment implications
            
        Visualization Logic:
        - Single numeric column: Distribution histogram
        - Multiple numeric columns: Correlation heatmap
        - Automatic chart selection based on data structure
        - Professional styling with matplotlib/seaborn
        
        Features:
        - Statistical summary generation
        - Automatic chart creation and display
        - Investment-focused narrative interpretation
        - Data quality and structure analysis
        
        Example:
            arjun.data_story(stock_df, "Monthly returns over 5 years")
        """
        # Basic data analysis
        summary_stats = df.describe()
        
        # Create simple visualization if numeric data exists
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            if len(numeric_cols) == 1:
                df[numeric_cols[0]].hist(bins=20, ax=ax)
                ax.set_title(f"Distribution of {numeric_cols[0]}")
            else:
                # Correlation heatmap for multiple numeric columns
                corr_matrix = df[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title("Correlation Matrix")
            plt.tight_layout()
            plt.show()
        
        # Ask Arjun to interpret the data
        data_context = f"""
        Data Summary:
        {summary_stats.to_string()}
        
        Chart description: {chart_description}
        
        Additional context:
        - Dataset shape: {df.shape}
        - Column types: {df.dtypes.to_dict()}
        """
        
        prompt = f"Looking at this financial/market data, what story does it tell? What insights and investment implications do you see? {data_context}"
        return self.ask(prompt)
    
    def compare_scenarios(self, scenario_a, scenario_b):
        """
        Compare two investment scenarios with balanced analysis.
        
        Provides structured comparison of investment options, strategies,
        or market conditions with pros, cons, and key considerations.
        
        Args:
            scenario_a (str): First investment scenario or option
            scenario_b (str): Second investment scenario or option
            
        Returns:
            Balanced comparative analysis with decision framework
            
        Example:
            arjun.compare_scenarios("Growth vs Value investing", "Bonds vs Stocks")
            arjun.compare_scenarios("US markets", "International diversification")
        """
        prompt = f"""
        Compare these two investment scenarios:
        
        Scenario A: {scenario_a}
        Scenario B: {scenario_b}
        
        Provide a balanced analysis of the pros, cons, and key considerations for each.
        """
        return self.ask(prompt)
    
    def show_conversation_history(self):
        """
        Display conversation history with professional formatting.
        
        Shows all previous Q&A exchanges in a clean, readable format
        with HTML styling. Truncates long responses for overview while
        maintaining readability.
        
        Features:
        - Professional HTML formatting with borders and styling
        - Numbered question sequence for easy reference
        - Response truncation for overview (200 chars + ellipsis)
        - Empty state handling with helpful message
        - Responsive design for notebook display
        
        Usage:
            arjun.show_conversation_history()
        """
        if not self.conversation_history:
            display(HTML('<p>No conversation history yet. Start by asking a question!</p>'))
            return
        
        html_content = """
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0;">
        <h3>📚 Conversation History</h3>
        """
        
        for i, exchange in enumerate(self.conversation_history, 1):
            html_content += f"""
            <div style="margin: 15px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
            <strong>Q{i}:</strong> {exchange['question']}<br>
            <strong>Arjun:</strong> {exchange['answer'][:200]}{'...' if len(exchange['answer']) > 200 else ''}
            </div>
            """
        
        html_content += "</div>"
        display(HTML(html_content))
    
    def clear_history(self):
        """
        Clear conversation history and reset session state.
        
        Removes all stored Q&A exchanges from memory and displays
        confirmation message. Useful for starting fresh analysis
        sessions or managing memory usage.
        
        Features:
        - Complete history clearing
        - Visual confirmation with HTML styling
        - Immediate effect (no confirmation required)
        
        Usage:
            arjun.clear_history()
        """
        self.conversation_history = []
        display(HTML('<p style="color: green;">✅ Conversation history cleared</p>'))
    
    def export_conversation(self, filename="arjun_conversation.txt"):
        """
        Export complete conversation history to a text file.
        
        Saves all Q&A exchanges to a formatted text file for documentation,
        sharing, or further analysis. Includes headers and separators for
        professional presentation.
        
        Args:
            filename (str): Output filename (default: "arjun_conversation.txt")
            
        Features:
        - Professional formatting with headers and separators
        - Complete response text (no truncation)
        - UTF-8 encoding for international character support
        - Numbered questions for easy reference
        - Success confirmation with HTML styling
        - Empty conversation handling
        
        Output Format:
        - File header with title
        - Numbered Q&A pairs
        - Visual separators between exchanges
        - Clean, readable text format
        
        Example:
            arjun.export_conversation("market_analysis_session.txt")
        """
        if not self.conversation_history:
            print("No conversation to export")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Arjun AI Investment Advisor - Conversation Export\n")
            f.write("=" * 50 + "\n\n")
            
            for i, exchange in enumerate(self.conversation_history, 1):
                f.write(f"Q{i}: {exchange['question']}\n")
                f.write(f"Arjun: {exchange['answer']}\n")
                f.write("-" * 30 + "\n\n")
        
        display(HTML(f'<p style="color: green;">✅ Conversation exported to {filename}</p>'))

# ═══════════════════════════ CONVENIENCE FUNCTIONS ═══════════════════════════ #

def create_arjun():
    """
    Create and return a new Arjun AI instance.
    
    Convenience function for quick instance creation without
    needing to specify the full class name and parameters.
    
    Returns:
        ArjunAI: Configured Arjun AI instance ready for use
        
    Example:
        arjun = create_arjun()
        arjun.ask("Your question here")
    """
    return ArjunAI()

def quick_ask(question):
    """
    Ask a quick question without creating a persistent instance.
    
    Useful for one-off questions where you don't need conversation
    history or ongoing interaction. Creates a temporary instance,
    gets the response, and returns plain text.
    
    Args:
        question (str): Investment question or analysis request
        
    Returns:
        str: Arjun's response as plain text (no markdown formatting)
        
    Example:
        response = quick_ask("What's your view on tech stocks?")
        print(response)
    """
    arjun = ArjunAI()
    return arjun.ask(question, show_markdown=False)

# ═══════════════════════════ DEMO AND EXAMPLES ═══════════════════════════ #

def demo_arjun():
    """
    Demonstrate Arjun AI capabilities with sample interactions.
    
    Runs through representative questions that showcase the model's
    distinctive voice characteristics and investment expertise.
    Perfect for new users to understand capabilities and voice style.
    
    Features Demonstrated:
    - Value investing philosophy
    - Analogical explanations
    - Historical crisis wisdom
    - Professional formatting
    - Conversation flow
    
    Usage:
        demo_arjun()  # Run in notebook cell
    """
    arjun = ArjunAI()
    
    display(HTML("""
    <h2>🧪 Arjun AI Demo</h2>
    <p>Here are some example interactions:</p>
    """))
    
    sample_questions = [
        "What's your view on value investing in the current market?",
        "Explain market volatility using an analogy",
        "What lessons from the 2008 crisis still apply today?"
    ]
    
    for question in sample_questions:
        arjun.ask(question)

# ═══════════════════════════ MODULE INITIALIZATION ═══════════════════════════ #

if __name__ == "__main__":
    print("🚀 Arjun AI Jupyter Helper loaded successfully!")
    print("═" * 60)
    print("📊 Usage: arjun = ArjunAI()")
    print("💬        arjun.ask('your investment question')")
    print("📈        arjun.analyze_data(df, 'analysis request')")
    print("🌍        arjun.emerging_markets_insight()")
    print("💡        arjun.explain_with_analogy('concept')")
    print("═" * 60)
    print("🎆 Ready for investment analysis and research!")
