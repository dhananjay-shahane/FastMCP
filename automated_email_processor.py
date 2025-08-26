#!/usr/bin/env python3
"""
Automated Email Processing Workflow
Continuously monitors emails from dhananjayshahane24@gmail.com and processes requests automatically
"""

import asyncio
import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("automated_email_processor")

# Load environment variables
load_dotenv()

# Import from our modules
from email_enhanced_server import email_handler, ollama_llm

class AutomatedEmailProcessor:
    """Automated email processing workflow"""
    
    def __init__(self):
        self.processed_emails = set()  # Track processed email IDs
        self.data_dir = Path("data")
        self.scripts_dir = Path("scripts")
        self.output_dir = Path("output")
        self.running = False
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.scripts_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
    async def start_monitoring(self, check_interval: int = 30):
        """Start continuous email monitoring"""
        logger.info("🚀 Starting automated email processing workflow...")
        logger.info(f"📧 Monitoring emails from: dhananjayshahane24@gmail.com")
        logger.info(f"⏰ Check interval: {check_interval} seconds")
        
        self.running = True
        
        while self.running:
            try:
                await self.process_new_emails()
                await asyncio.sleep(check_interval)
            except KeyboardInterrupt:
                logger.info("❌ Stopping email processor...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(check_interval)
    
    async def process_new_emails(self):
        """Check for and process new emails"""
        try:
            # Get recent emails (last 5)
            logger.info("📨 Checking for new emails...")
            emails = await email_handler.get_filtered_emails(limit=5)
            
            if not emails:
                logger.debug("No new emails found")
                return
                
            # Process each unprocessed email
            for email_data in emails:
                email_id = email_data.get('id')
                if email_id not in self.processed_emails:
                    await self.process_single_email(email_data)
                    self.processed_emails.add(email_id)
                    
        except Exception as e:
            logger.error(f"Error processing emails: {str(e)}")
    
    async def process_single_email(self, email_data: Dict[str, Any]):
        """Process a single email using MCP tools and intelligent LLM responses"""
        try:
            sender = email_data.get('from', '')
            subject = email_data.get('subject', '')
            body = email_data.get('body', '')
            
            logger.info(f"📧 Processing email: {subject[:50]}...")
            logger.info(f"From: {sender}")
            
            # Extract sender email
            sender_email = self.extract_email_address(sender)
            if not sender_email:
                logger.warning(f"Could not extract sender email from: {sender}")
                return
            
            # Use MCP tool to analyze email request with LLM  
            logger.info(f"🤖 Analyzing email with LLM...")
            analysis_result = await self.analyze_email_with_mcp_tools(sender_email, subject, body)
            
            # Generate response based on analysis
            if analysis_result['needs_clarification']:
                # Ask for more details
                response_text = await self.generate_clarification_request(analysis_result, sender_email, subject)
                attachments = []
            elif analysis_result['is_analysis_request']:
                # Execute data analysis and generate response
                response_text, attachments = await self.execute_analysis_workflow(analysis_result, sender_email, subject, body)
            else:
                # General response with available options
                response_text = await self.generate_general_response(analysis_result, sender_email, subject, body)
                attachments = []
            
            # Send response email
            response_subject = f"Re: {subject}" if not subject.startswith('Re:') else subject
            
            logger.info(f"📤 Sending intelligent response to: {sender_email}")
            success = await email_handler.send_email(
                to_email=sender_email,
                subject=response_subject,
                body=response_text,
                attachments=attachments
            )
            
            if success:
                logger.info(f"✅ Response sent successfully to {sender_email}")
                if attachments:
                    logger.info(f"📎 Included {len(attachments)} attachments")
            else:
                logger.error(f"❌ Failed to send response to {sender_email}")
                
        except Exception as e:
            logger.error(f"Error processing email: {str(e)}")
    
    def extract_email_address(self, from_field: str) -> Optional[str]:
        """Extract email address from From field"""
        try:
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', from_field)
            return email_match.group(0) if email_match else None
        except Exception:
            return None
    
    async def analyze_email_with_mcp_tools(self, sender_email: str, subject: str, body: str) -> Dict[str, Any]:
        """Use MCP tools to analyze email with LLM intelligence"""
        try:
            # Import from the server module
            import email_enhanced_server as server
            
            # Get comprehensive data analysis
            data_summary = await server.get_data_summary()
            
            # Analyze email with LLM using the server's ollama_llm directly
            analysis = await server.analyze_email_request(body, sender_email, subject)
            
            if not analysis['success']:
                return {
                    'is_analysis_request': False,
                    'needs_clarification': True,
                    'error': analysis.get('error', 'Analysis failed')
                }
            
            request_analysis = analysis['request_analysis']
            
            return {
                'is_analysis_request': request_analysis['intent'] == 'visualization_request',
                'needs_clarification': request_analysis['needs_clarification'],
                'clarification_needed': request_analysis.get('clarification_needed', []),
                'visualization_type': request_analysis.get('visualization_type'),
                'data_file_mentioned': request_analysis.get('data_file_mentioned'),
                'available_data': data_summary,
                'llm_response': analysis['llm_response'],
                'original_analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing email with MCP tools: {str(e)}")
            return {
                'is_analysis_request': False,
                'needs_clarification': True,
                'error': str(e)
            }
    
    async def generate_clarification_request(self, analysis_result: Dict[str, Any], sender_email: str, subject: str) -> str:
        """Generate clarification request using LLM"""
        try:
            from email_enhanced_server import generate_intelligent_response
            
            clarification_context = f"""
The user's request needs clarification. Missing information:
{', '.join(analysis_result.get('clarification_needed', []))}

Available data files: {[f['name'] for f in analysis_result.get('available_data', {}).get('csv_files', [])]}

Generate a helpful response asking for the missing details to create their visualization.
"""
            
            response = await generate_intelligent_response(
                clarification_context, sender_email, subject
            )
            
            if response['success']:
                return response['response_content']
            else:
                # Fallback clarification
                available_files = [f['name'] for f in analysis_result.get('available_data', {}).get('csv_files', [])]
                return f"""Hello!

I'd be happy to help you create a visualization! However, I need a bit more information:

{chr(10).join(f'• {item}' for item in analysis_result.get('clarification_needed', []))}

Here's what I have available:
📁 Data files: {', '.join(available_files)}
📊 Chart types: Bar charts, Line graphs, Pie charts, Statistical analysis

Example request: "Create a bar chart from sales.csv showing sales_amount by category"

Best regards,
Data Analysis Assistant"""
                
        except Exception as e:
            logger.error(f"Error generating clarification: {str(e)}")
            return "Hello! I need more details about your data analysis request. Please specify the data file and type of visualization you'd like."
    
    async def execute_analysis_workflow(self, analysis_result: Dict[str, Any], sender_email: str, subject: str, body: str) -> tuple:
        """Execute complete data analysis workflow using MCP tools"""
        try:
            from email_enhanced_server import execute_data_analysis, generate_intelligent_response
            
            # Determine script and data file
            viz_type = analysis_result.get('visualization_type', 'bar_chart')
            data_file = analysis_result.get('data_file_mentioned')
            
            # Default to first available file if none specified
            if not data_file:
                available_files = analysis_result.get('available_data', {}).get('csv_files', [])
                if available_files:
                    data_file = available_files[0]['name']
                else:
                    return "I couldn't find any data files to analyze.", []
            
            # Map visualization type to script
            script_map = {
                'bar_chart': 'bar_chart.py',
                'line_graph': 'line_graph.py', 
                'pie_chart': 'pie_chart.py',
                'statistics': 'custom_stats.py'
            }
            
            script_name = script_map.get(viz_type, 'bar_chart.py')
            
            logger.info(f"⚙️ Executing {script_name} with {data_file}")
            
            # Execute analysis using MCP tool
            execution_result = await execute_data_analysis(script_name, data_file)
            
            if execution_result['success']:
                # Generate intelligent response with results
                analysis_context = f"""
Successfully completed data analysis:
- Script: {script_name}
- Data file: {data_file}
- Generated files: {execution_result.get('generated_files', [])}
- Output: {execution_result.get('script_output', '')}

Create a professional response mentioning the successful analysis and attached visualizations.
"""
                
                response = await generate_intelligent_response(
                    analysis_context, sender_email, subject, execution_result
                )
                
                if response['success']:
                    response_text = response['response_content']
                else:
                    response_text = f"""Hello!

I've successfully completed your data analysis request!

📊 Analysis: {viz_type.replace('_', ' ').title()}
📁 Data file: {data_file}
🖼️ Generated: {len(execution_result.get('generated_files', []))} visualization(s)

The results are attached as PNG files. Each file includes a timestamp for easy identification.

Best regards,
Data Analysis Assistant
Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
                
                return response_text, execution_result.get('generated_files', [])
            else:
                error_response = f"""Hello!

I encountered an issue while processing your analysis request:

Error: {execution_result.get('error', 'Unknown error')}

Please check:
• Data file exists: {data_file}
• Column names are correct
• File format is valid CSV

Available files: {', '.join([f['name'] for f in analysis_result.get('available_data', {}).get('csv_files', [])])}

Best regards,
Data Analysis Assistant"""
                
                return error_response, []
                
        except Exception as e:
            logger.error(f"Error in analysis workflow: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}", []
    
    async def generate_general_response(self, analysis_result: Dict[str, Any], sender_email: str, subject: str, body: str) -> str:
        """Generate general response using LLM"""
        try:
            from email_enhanced_server import generate_intelligent_response
            
            response = await generate_intelligent_response(body, sender_email, subject)
            
            if response['success']:
                return response['response_content']
            else:
                # Fallback response
                available_files = [f['name'] for f in analysis_result.get('available_data', {}).get('csv_files', [])]
                return f"""Hello!

Thank you for your email. I'm an AI assistant specializing in data analysis and visualization.

🔧 My capabilities:
• Create bar charts, line graphs, and pie charts
• Generate statistical analysis reports
• Process CSV data files
• Deliver timestamped PNG visualizations

📁 Available data: {', '.join(available_files)}

For analysis requests, please specify:
1. Data file name
2. Type of visualization (bar, line, pie, stats)
3. Columns to analyze

Example: "Create a bar chart from sales.csv showing revenue by region"

Best regards,
Data Analysis Assistant"""
                
        except Exception as e:
            logger.error(f"Error generating general response: {str(e)}")
            return "Thank you for your email! I'm here to help with data analysis and visualization. Please let me know what you'd like to analyze."
    
    # Removed - now using MCP tools for analysis execution

async def main():
    """Main function to start automated email processing"""
    try:
        processor = AutomatedEmailProcessor()
        
        print("🚀 AUTOMATED EMAIL PROCESSOR STARTING...")
        print("📧 Monitoring emails from: dhananjayshahane24@gmail.com")
        print("🔄 Will process data analysis requests automatically")
        print("📊 Available: bar charts, line graphs, pie charts, statistics")
        print("💾 Output: PNG files sent as email attachments")
        print("\nPress Ctrl+C to stop...")
        
        await processor.start_monitoring(check_interval=30)  # Check every 30 seconds
        
    except KeyboardInterrupt:
        print("\n✅ Email processor stopped by user")
    except Exception as e:
        print(f"❌ Error in email processor: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())