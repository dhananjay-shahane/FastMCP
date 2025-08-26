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
        """Process a single email and generate response"""
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
            
            # Use LLM to understand the request
            context = f"""
Available CSV files: {list(self.data_dir.glob('*.csv'))}
Available scripts: {list(self.scripts_dir.glob('*.py'))}
System can generate: bar charts, line graphs, pie charts, statistical analysis
Output format: PNG files with timestamps
"""
            
            # Get LLM response to understand intent
            llm_response = await ollama_llm.generate_response(body, context)
            
            # Parse the email for data analysis requests
            analysis_request = self.parse_analysis_request(body)
            
            response_text = ""
            attachments = []
            
            if analysis_request['is_analysis_request']:
                logger.info(f"🔍 Analysis request detected: {analysis_request['request_type']}")
                
                # Execute the analysis
                execution_result = await self.execute_analysis(analysis_request, body)
                
                if execution_result['success']:
                    response_text = f"""Hello! I've processed your data analysis request.

Request: {subject}

{llm_response}

Analysis Results:
{execution_result.get('output', '')}

Generated files: {', '.join([Path(f).name for f in execution_result.get('attachments', [])])}

The analysis has been completed and the results are attached as PNG files.

Best regards,
MCP Data Analysis Server
Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
                    
                    attachments = execution_result.get('attachments', [])
                else:
                    response_text = f"""Hello! I received your data analysis request but encountered an issue.

Request: {subject}

Error: {execution_result.get('error', 'Unknown error')}

{llm_response}

Please check your request and make sure:
- The CSV file name is correct
- Column names exist in the data
- The request format is clear

Best regards,
MCP Data Analysis Server"""
            else:
                # General response
                response_text = f"""Hello! Thank you for your email.

{llm_response}

If you need data analysis, please specify:
- CSV file name (from data directory)
- Type of chart (bar, line, pie)
- Column names to analyze

Best regards,
MCP Data Analysis Server"""
            
            # Send response email
            response_subject = f"Re: {subject}" if not subject.startswith('Re:') else subject
            
            logger.info(f"📤 Sending response to: {sender_email}")
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
    
    def parse_analysis_request(self, email_body: str) -> Dict[str, Any]:
        """Parse email for data analysis requests"""
        try:
            request_info = {
                'is_analysis_request': False,
                'request_type': 'unknown',
                'data_file': None,
                'chart_type': None,
                'columns': [],
                'parameters': {}
            }
            
            body_lower = email_body.lower()
            
            # Check for analysis keywords
            analysis_keywords = [
                'analyze', 'analysis', 'chart', 'graph', 'plot', 'visualization',
                'statistics', 'stats', 'report', 'dashboard', 'data', 'csv'
            ]
            
            if any(keyword in body_lower for keyword in analysis_keywords):
                request_info['is_analysis_request'] = True
            
            # Determine chart type
            if 'bar' in body_lower:
                request_info['chart_type'] = 'bar_chart'
                request_info['request_type'] = 'bar_chart'
            elif 'line' in body_lower or 'trend' in body_lower:
                request_info['chart_type'] = 'line_graph'
                request_info['request_type'] = 'line_graph'
            elif 'pie' in body_lower:
                request_info['chart_type'] = 'pie_chart'
                request_info['request_type'] = 'pie_chart'
            elif 'stat' in body_lower:
                request_info['chart_type'] = 'custom_stats'
                request_info['request_type'] = 'custom_stats'
            
            # Extract CSV file names
            csv_files = re.findall(r'(\w+\.csv)', email_body, re.IGNORECASE)
            if csv_files:
                request_info['data_file'] = csv_files[0]
            
            return request_info
            
        except Exception as e:
            logger.error(f"Error parsing analysis request: {str(e)}")
            return {'is_analysis_request': False, 'error': str(e)}
    
    async def execute_analysis(self, request: Dict[str, Any], email_body: str) -> Dict[str, Any]:
        """Execute the data analysis request"""
        try:
            data_file = request.get('data_file')
            chart_type = request.get('chart_type', 'bar_chart')
            
            # Default to first available CSV file if none specified
            if not data_file:
                csv_files = list(self.data_dir.glob('*.csv'))
                if csv_files:
                    data_file = csv_files[0].name
                else:
                    return {
                        'success': False,
                        'error': 'No CSV files found in data directory'
                    }
            
            # Check if data file exists
            data_path = self.data_dir / data_file
            if not data_path.exists():
                return {
                    'success': False,
                    'error': f'CSV file {data_file} not found in data directory'
                }
            
            # Determine script to run
            script_map = {
                'bar_chart': 'bar_chart.py',
                'line_graph': 'line_graph.py',
                'pie_chart': 'pie_chart.py',
                'custom_stats': 'custom_stats.py'
            }
            
            script_name = script_map.get(chart_type, 'bar_chart.py')
            script_path = self.scripts_dir / script_name
            
            if not script_path.exists():
                return {
                    'success': False,
                    'error': f'Script {script_name} not found in scripts directory'
                }
            
            # Execute the script
            logger.info(f"⚙️ Executing {script_name} with {data_file}")
            
            try:
                result = subprocess.run([
                    sys.executable, str(script_path), str(data_path)
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    output = result.stdout
                    
                    # Find generated PNG files
                    timestamp_pattern = datetime.now().strftime('%Y%m%d_%H%M')
                    png_files = list(self.output_dir.glob(f'*{timestamp_pattern[:8]}*.png'))
                    
                    # If no files with exact timestamp, get most recent PNG files
                    if not png_files:
                        png_files = sorted(self.output_dir.glob('*.png'), key=os.path.getmtime, reverse=True)[:3]
                    
                    return {
                        'success': True,
                        'output': output,
                        'script_name': script_name,
                        'data_file': data_file,
                        'attachments': [str(f) for f in png_files]
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Script execution failed: {result.stderr}'
                    }
                    
            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'error': 'Script execution timed out'
                }
                
        except Exception as e:
            logger.error(f"Error executing analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

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