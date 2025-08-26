#!/usr/bin/env python3
"""
Enhanced MCP Server with Email IMAP Filtering and Ollama LLM Integration
"""

import os
import subprocess
import logging
import sys
import asyncio
import aiohttp
import json
import imaplib
import smtplib
import ssl
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Email related imports - use same pattern as working server.py
import email as email_module
from email.mime.text import MIMEText as MimeText
from email.mime.multipart import MIMEMultipart as MimeMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.header import decode_header
import mimetypes

try:
    from fastmcp import FastMCP, Context
except ImportError:
    print("Error: FastMCP not available. Install with: pip install fastmcp")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("email_mcp_server")

# Initialize FastMCP server
mcp = FastMCP("EmailDataVizServer")

# Get absolute paths
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / "data"
SCRIPTS_DIR = SCRIPT_DIR / "scripts"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOGS_DIR = SCRIPT_DIR / "logs"

# Load environment variables
load_dotenv()

# Email and LLM Configuration from .env file
EMAIL_CONFIG = {
    'smtp_server': os.getenv('MAIL_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('MAIL_PORT', '587')),
    'imap_server': 'imap.gmail.com',
    'imap_port': 993,
    'username': os.getenv('EMAIL_USER'),
    'password': os.getenv('EMAIL_PASSWORD'),
    'allowed_sender': 'dhananjayshahane24@gmail.com'  # Filter emails only from this address
}

OLLAMA_CONFIG = {
    'base_url': os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434'),
    'model': 'llama3.2:1b',  # Using the specific model requested
    'timeout': 30
}

class EmailHandler:
    """Handle email operations with IMAP filtering and SMTP sending"""
    
    def __init__(self, config: dict):
        self.config = config
        self.smtp_server = None
        self.imap_server = None
    
    async def connect_imap(self):
        """Connect to IMAP server"""
        try:
            self.imap_server = imaplib.IMAP4_SSL(self.config['imap_server'], self.config['imap_port'])
            self.imap_server.login(self.config['username'], self.config['password'])
            logger.info("Successfully connected to IMAP server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IMAP: {str(e)}")
            return False
    
    async def get_filtered_emails(self, folder='INBOX', limit=10):
        """Get emails only from allowed sender"""
        try:
            if not self.imap_server:
                connected = await self.connect_imap()
                if not connected or not self.imap_server:
                    return []
            
            self.imap_server.select(folder)
            
            # Search for emails from specific sender
            search_criteria = f'FROM "{self.config["allowed_sender"]}"'
            status, messages = self.imap_server.search(None, search_criteria)
            
            if status != 'OK':
                return []
            
            email_ids = messages[0].split()
            emails = []
            
            # Get latest emails (limited)
            for email_id in email_ids[-limit:]:
                status, msg_data = self.imap_server.fetch(email_id, '(RFC822)')
                if (status == 'OK' and msg_data and isinstance(msg_data, list) and 
                    len(msg_data) > 0 and msg_data[0] and isinstance(msg_data[0], tuple) and 
                    len(msg_data[0]) > 1 and isinstance(msg_data[0][1], bytes)):
                    email_message = email_module.message_from_bytes(msg_data[0][1])
                    
                    # Decode subject
                    subject = decode_header(email_message["Subject"])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode()
                    
                    # Get email body
                    body = self._get_email_body(email_message)
                    
                    emails.append({
                        'id': email_id.decode(),
                        'from': email_message["From"],
                        'subject': subject,
                        'date': email_message["Date"],
                        'body': body
                    })
            
            logger.info(f"Retrieved {len(emails)} emails from {self.config['allowed_sender']}")
            return emails
            
        except Exception as e:
            logger.error(f"Error retrieving emails: {str(e)}")
            return []
    
    def _get_email_body(self, email_message):
        """Extract email body content"""
        try:
            if email_message.is_multipart():
                for part in email_message.walk():
                    if part.get_content_type() == "text/plain":
                        return part.get_payload(decode=True).decode()
            else:
                return email_message.get_payload(decode=True).decode()
        except Exception as e:
            logger.error(f"Error extracting email body: {str(e)}")
            return ""
    
    async def send_email(self, to_email: str, subject: str, body: str, attachments=None):
        """Send email response with optional attachments"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.config['username']
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            # Add attachments if provided
            if attachments:
                for attachment_path in attachments:
                    if Path(attachment_path).exists():
                        self._attach_file(msg, attachment_path)
                        logger.info(f"Added attachment: {attachment_path}")
                    else:
                        logger.warning(f"Attachment not found: {attachment_path}")
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['username'], self.config['password'])
            server.send_message(msg)
            server.quit()
            
            attachment_info = f" with {len(attachments)} attachments" if attachments else ""
            logger.info(f"Email sent successfully to {to_email}{attachment_info}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def _attach_file(self, msg, file_path):
        """Attach file to email message"""
        try:
            # Guess the content type based on the file's extension
            ctype, encoding = mimetypes.guess_type(file_path)
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'
            
            maintype, subtype = ctype.split('/', 1)
            
            with open(file_path, 'rb') as fp:
                attachment = MIMEBase(maintype, subtype)
                attachment.set_payload(fp.read())
                encoders.encode_base64(attachment)
                
                # Add header with filename
                filename = Path(file_path).name
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}'
                )
                msg.attach(attachment)
                
        except Exception as e:
            logger.error(f"Failed to attach file {file_path}: {str(e)}")

class OllamaLLM:
    """Handle Ollama LLM interactions"""
    
    def __init__(self, config: dict):
        self.config = config
        self.base_url = config['base_url']
        self.model = config['model']
        self.timeout = config['timeout']
    
    async def is_available(self):
        """Check if Ollama server is available"""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=timeout) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Ollama server not available: {str(e)}")
            return False
    
    async def generate_response(self, prompt: str, context: str = ""):
        """Generate response using Ollama"""
        try:
            full_prompt = f"""You are an AI assistant for a comprehensive data analysis MCP server system. 

{context}

Email Content: {prompt}

Instructions:
- Be professional and helpful
- If they ask for data analysis, offer specific visualizations (bar charts, line graphs, pie charts)
- Mention that you can generate timestamped PNG files in the output directory
- If they provide data or ask about specific datasets, offer relevant analysis
- Keep responses concise but informative
- Always offer to help with next steps

Please provide a helpful response:"""

            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', 'Sorry, I could not generate a response.')
                    else:
                        logger.error(f"Ollama API error: {response.status}")
                        return "Sorry, there was an error generating the response."
                        
        except Exception as e:
            logger.warning(f"Error generating Ollama response: {str(e)}")
            # Fallback response when Ollama is unavailable
            return f"""Thank you for your email! 

I'm an AI assistant for data analysis and visualization. I can help you with:

📊 **Data Visualization Services:**
- Bar charts, line graphs, and pie charts from CSV data
- High-quality PNG outputs with timestamps
- Statistical analysis and insights

📁 **Currently Available:**
- CSV files in the data directory
- Python analysis scripts (bar_chart.py, line_graph.py, pie_chart.py)
- Automated report generation

To use these services, please:
1. Upload your CSV data files
2. Specify what type of visualization you need
3. I'll generate timestamped PNG files for you

The system will be back to full AI-powered responses once Ollama is running locally. For immediate assistance, please specify your data analysis needs.

Best regards,
Your Data Analysis Assistant"""

# Initialize handlers
email_handler = EmailHandler(EMAIL_CONFIG)
ollama_llm = OllamaLLM(OLLAMA_CONFIG)

# Resources - CSV files and scripts (keeping existing functionality)
@mcp.resource("file://data/{filename}")
async def read_csv_resource(filename: str) -> str:
    """Read CSV files from the data directory"""
    try:
        data_path = DATA_DIR / filename
        if not data_path.exists():
            raise FileNotFoundError(f"CSV file {filename} not found in {DATA_DIR}")
        
        if not data_path.suffix.lower() == '.csv':
            raise ValueError(f"File {filename} is not a CSV file")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"Successfully read CSV resource: {filename}")
        return content
        
    except Exception as e:
        logger.error(f"Error reading CSV resource {filename}: {str(e)}")
        raise

@mcp.resource("file://scripts/{filename}")
async def read_script_resource(filename: str) -> str:
    """Read Python scripts from the scripts directory"""
    try:
        script_path = SCRIPTS_DIR / filename
        if not script_path.exists():
            raise FileNotFoundError(f"Script file {filename} not found in {SCRIPTS_DIR}")
        
        if not filename.endswith('.py'):
            raise ValueError(f"File {filename} is not a Python script")
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"Successfully read script resource: {filename}")
        return content
        
    except Exception as e:
        logger.error(f"Error reading script resource {filename}: {str(e)}")
        raise

# Email Tools
@mcp.tool()
async def check_filtered_emails(limit: int = 5, ctx: Context | None = None) -> Dict[str, Any]:
    """Check for new emails from allowed sender only"""
    try:
        emails = await email_handler.get_filtered_emails(limit=limit)
        
        logger.info(f"Checked emails: found {len(emails)} from {EMAIL_CONFIG['allowed_sender']}")
        return {
            "success": True,
            "emails_found": len(emails),
            "allowed_sender": EMAIL_CONFIG['allowed_sender'],
            "emails": emails
        }
        
    except Exception as e:
        logger.error(f"Error checking emails: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to check emails: {str(e)}"
        }

@mcp.tool()
async def process_email_with_llm(email_content: str, sender: str, subject: str, ctx: Context | None = None) -> Dict[str, Any]:
    """Process email content with Ollama LLM and generate response"""
    try:
        # Check if Ollama is available
        if not await ollama_llm.is_available():
            return {
                "success": False,
                "error": "Ollama LLM server is not available"
            }
        
        # Generate response using Ollama
        context = f"Email from: {sender}, Subject: {subject}"
        llm_response = await ollama_llm.generate_response(email_content, context)
        
        logger.info(f"Generated LLM response for email from {sender}")
        return {
            "success": True,
            "original_email": {
                "sender": sender,
                "subject": subject,
                "content": email_content
            },
            "llm_response": llm_response,
            "model_used": OLLAMA_CONFIG['model']
        }
        
    except Exception as e:
        logger.error(f"Error processing email with LLM: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to process email: {str(e)}"
        }

@mcp.tool()
async def send_automated_reply(to_email: str, subject: str, response_content: str, ctx: Context | None = None) -> Dict[str, Any]:
    """Send automated reply email"""
    try:
        # Verify recipient is the allowed sender
        if to_email != EMAIL_CONFIG['allowed_sender']:
            return {
                "success": False,
                "error": f"Can only send replies to {EMAIL_CONFIG['allowed_sender']}"
            }
        
        # Send email
        reply_subject = f"Re: {subject}" if not subject.startswith("Re:") else subject
        success = await email_handler.send_email(to_email, reply_subject, response_content)
        
        if success:
            logger.info(f"Automated reply sent to {to_email}")
            return {
                "success": True,
                "recipient": to_email,
                "subject": reply_subject,
                "message": "Reply sent successfully"
            }
        else:
            return {
                "success": False,
                "error": "Failed to send email"
            }
        
    except Exception as e:
        logger.error(f"Error sending automated reply: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to send reply: {str(e)}"
        }

@mcp.tool()
async def check_and_respond_to_emails(ctx: Context | None = None) -> Dict[str, Any]:
    """Complete workflow: Check emails, process with LLM, generate charts, and send responses with attachments"""
    try:
        # Check for new emails
        emails = await email_handler.get_filtered_emails(limit=3)
        
        if not emails:
            return {
                "success": True,
                "message": "No new emails to process",
                "processed_count": 0
            }
        
        processed_emails = []
        
        for email_data in emails:
            try:
                # Analyze email content to determine if visualization is needed
                email_content = email_data['body'].lower()
                attachments = []
                charts_generated = []
                
                # Enhanced LLM processing with context about available tools
                context = f"""You are an AI assistant for a data analysis MCP server. You can:
                1. Execute Python scripts for data visualization (bar charts, line graphs, pie charts)
                2. Analyze CSV data files in the data directory
                3. Generate timestamped PNG visualizations in the output directory
                4. Provide statistical analysis and insights
                
                Available CSV files: {await get_available_csv_files()}
                Available scripts: {await get_available_scripts()}
                
                Email from: {email_data['from']}
                Subject: {email_data['subject']}
                
                The user is asking: {email_data['body']}
                
                If they want data analysis or charts, I will generate the visualization and attach it to the email.
                Please provide a helpful response mentioning the attached chart if one is generated."""
                
                # Generate visualizations based on email content
                if any(keyword in email_content for keyword in ['sales', 'chart', 'graph', 'data', 'visualization', 'report', 'cash flow', 'traffic']):
                    try:
                        import subprocess
                        import os
                        
                        # Determine best chart type based on request
                        if 'sales' in email_content or 'category' in email_content:
                            # Generate sales bar chart
                            result = subprocess.run([
                                'python3', str(SCRIPTS_DIR / 'bar_chart.py'),
                                str(DATA_DIR / 'sales.csv'), 'category', 'sales_amount',
                                '--title', 'Sales Analysis by Category - Email Response'
                            ], capture_output=True, text=True)
                            
                        elif 'traffic' in email_content or 'trend' in email_content:
                            # Generate line graph for trends
                            result = subprocess.run([
                                'python3', str(SCRIPTS_DIR / 'line_graph.py'),
                                str(DATA_DIR / 'trends.csv'), 'date', 'value',
                                '--title', 'Traffic/Trends Analysis - Email Response'
                            ], capture_output=True, text=True)
                            
                        elif 'cash flow' in email_content or 'financial' in email_content:
                            # Generate pie chart for sales distribution
                            result = subprocess.run([
                                'python3', str(SCRIPTS_DIR / 'pie_chart.py'),
                                str(DATA_DIR / 'sales.csv'), 'region', 'sales_amount',
                                '--title', 'Cash Flow by Region - Email Response'
                            ], capture_output=True, text=True)
                        else:
                            # Default: sales bar chart
                            result = subprocess.run([
                                'python3', str(SCRIPTS_DIR / 'bar_chart.py'),
                                str(DATA_DIR / 'sales.csv'), 'category', 'sales_amount',
                                '--title', 'Data Analysis - Email Response'
                            ], capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            # Extract output file path
                            output_lines = result.stdout.split('\n')
                            for line in output_lines:
                                if 'File:' in line and '.png' in line:
                                    chart_path = line.replace('File:', '').strip()
                                    if os.path.exists(chart_path):
                                        attachments.append(chart_path)
                                        charts_generated.append(os.path.basename(chart_path))
                                        logger.info(f"Generated chart: {chart_path}")
                                    break
                        else:
                            logger.error(f"Chart generation failed: {result.stderr}")
                            
                    except Exception as chart_error:
                        logger.error(f"Error generating chart: {str(chart_error)}")
                
                # Generate LLM response
                llm_response = await ollama_llm.generate_response(
                    email_data['body'], context
                )
                
                # Add attachment info to response if charts were generated
                if attachments:
                    chart_names = ', '.join(charts_generated)
                    llm_response += f"\n\n📊 **Attached Analysis:**\nI've generated and attached the requested data visualization: `{chart_names}`. This chart provides a professional analysis of your data with timestamped formatting for your records.\n\nThe visualization is ready for your review and can be used in presentations or reports."
                
                # Send reply with attachments
                reply_sent = await email_handler.send_email(
                    EMAIL_CONFIG['allowed_sender'],
                    f"Re: {email_data['subject']}",
                    llm_response,
                    attachments=attachments
                )
                
                processed_emails.append({
                    "email_id": email_data['id'],
                    "subject": email_data['subject'],
                    "reply_sent": reply_sent,
                    "response_preview": llm_response[:150] + "...",
                    "full_response": llm_response,
                    "attachments": attachments,
                    "charts_generated": len(attachments)
                })
                
            except Exception as email_error:
                logger.error(f"Error processing individual email: {str(email_error)}")
                continue
        
        total_charts = sum(email.get('charts_generated', 0) for email in processed_emails)
        logger.info(f"Processed {len(processed_emails)} emails with {total_charts} chart attachments")
        return {
            "success": True,
            "processed_count": len(processed_emails),
            "total_charts_generated": total_charts,
            "processed_emails": processed_emails,
            "allowed_sender": EMAIL_CONFIG['allowed_sender']
        }
        
    except Exception as e:
        logger.error(f"Error in automated email workflow: {str(e)}")
        return {
            "success": False,
            "error": f"Workflow failed: {str(e)}"
        }

# System Tools
@mcp.tool()
async def check_system_status(ctx: Context | None = None) -> Dict[str, Any]:
    """Check system status including email and LLM connectivity"""
    try:
        # Check Ollama availability
        ollama_available = await ollama_llm.is_available()
        
        # Check email connection
        email_connected = await email_handler.connect_imap()
        
        # Check directories
        data_exists = DATA_DIR.exists()
        scripts_exists = SCRIPTS_DIR.exists()
        
        return {
            "success": True,
            "system_status": {
                "ollama_llm_available": ollama_available,
                "ollama_url": OLLAMA_CONFIG['base_url'],
                "ollama_model": OLLAMA_CONFIG['model'],
                "email_imap_connected": email_connected,
                "email_allowed_sender": EMAIL_CONFIG['allowed_sender'],
                "data_directory_exists": data_exists,
                "scripts_directory_exists": scripts_exists
            },
            "directories": {
                "script_dir": str(SCRIPT_DIR),
                "data_dir": str(DATA_DIR),
                "scripts_dir": str(SCRIPTS_DIR)
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking system status: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to check system status: {str(e)}"
        }

# Helper functions for email processing
async def get_available_csv_files() -> List[str]:
    """Get list of available CSV files"""
    try:
        if DATA_DIR.exists():
            return [f.name for f in DATA_DIR.glob("*.csv")]
        return []
    except:
        return []

async def get_available_scripts() -> List[str]:
    """Get list of available Python scripts"""
    try:
        if SCRIPTS_DIR.exists():
            return [f.name for f in SCRIPTS_DIR.glob("*.py")]
        return []
    except:
        return []

# Enhanced script execution tools
@mcp.tool()
async def execute_script(script_name: str, args: List[str] = [], ctx: Context | None = None) -> Dict[str, Any]:
    """Execute Python scripts dynamically and return results"""
    try:
        if args is None:
            args = []
            
        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            return {
                "success": False,
                "error": f"Script {script_name} not found in {SCRIPTS_DIR}"
            }
        
        if not script_name.endswith('.py'):
            return {
                "success": False,
                "error": f"File {script_name} is not a Python script"
            }
        
        cmd = ["python", str(script_path)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(SCRIPT_DIR)
        )
        
        if result.returncode != 0:
            logger.error(f"Script execution failed: {result.stderr}")
            return {
                "success": False,
                "error": f"Script execution failed: {result.stderr}",
                "return_code": result.returncode
            }
        
        logger.info(f"Successfully executed script: {script_name}")
        return {
            "success": True,
            "output": result.stdout,
            "script_name": script_name,
            "args": args
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Script execution timed out (5 minutes)"
        }
    except Exception as e:
        logger.error(f"Error executing script {script_name}: {str(e)}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

@mcp.tool()
async def list_available_resources(ctx: Context | None = None) -> Dict[str, Any]:
    """List all available CSV files and Python scripts"""
    try:
        csv_files = []
        script_files = []
        
        if DATA_DIR.exists():
            csv_files = [f.name for f in DATA_DIR.glob("*.csv")]
        
        if SCRIPTS_DIR.exists():
            script_files = [f.name for f in SCRIPTS_DIR.glob("*.py")]
        
        return {
            "success": True,
            "csv_files": csv_files,
            "script_files": script_files,
            "total_csv": len(csv_files),
            "total_scripts": len(script_files),
            "email_config": {
                "allowed_sender": EMAIL_CONFIG['allowed_sender'],
                "email_account": EMAIL_CONFIG['username']
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing resources: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to list resources: {str(e)}"
        }

# Additional MCP tools for enhanced email processing
@mcp.tool()
async def process_email_and_execute_analysis(email_content: str, csv_file: str = "", analysis_type: str = "summary", ctx: Context | None = None) -> Dict[str, Any]:
    """Process email request and execute data analysis if requested"""
    try:
        # First understand the email with LLM
        context = f"""Available CSV files: {await get_available_csv_files()}
        Available scripts: bar_chart.py, line_graph.py, pie_chart.py, simple_stats.py
        Analysis types: bar_chart, line_graph, pie_chart, statistics
        
        Determine if this email is requesting data analysis and what type."""
        
        llm_analysis = await ollama_llm.generate_response(email_content, context)
        
        result = {
            "success": True,
            "email_analysis": llm_analysis,
            "analysis_executed": False,
            "output_files": []
        }
        
        # If CSV file and analysis type provided, execute it
        if csv_file and analysis_type in ['bar_chart', 'line_graph', 'pie_chart']:
            script_name = f"{analysis_type}.py"
            execution_result = await execute_script(script_name, [csv_file])
            
            if execution_result.get('success'):
                result["analysis_executed"] = True
                result["execution_output"] = execution_result.get('output', '')
                # Extract output file from execution result
                output_lines = execution_result.get('output', '').split('\n')
                for line in output_lines:
                    if 'File:' in line and '.png' in line:
                        result["output_files"].append(line.replace('File:', '').strip())
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing email analysis: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to process analysis: {str(e)}"
        }

@mcp.tool()
async def smart_email_response(email_content: str, sender: str, subject: str, auto_send: bool = False, ctx: Context | None = None) -> Dict[str, Any]:
    """Generate intelligent email response with potential data analysis"""
    try:
        # Get available resources
        csv_files = await get_available_csv_files()
        scripts = await get_available_scripts()
        
        # Enhanced context for LLM
        context = f"""You are responding to an email about data analysis services.
        
        Available CSV files: {', '.join(csv_files)}
        Available analysis scripts: {', '.join(scripts)}
        
        You can:
        1. Generate bar charts, line graphs, and pie charts from CSV data
        2. Provide statistical analysis and insights
        3. Create timestamped PNG visualizations saved to output directory
        4. Process data from the available CSV files
        
        Email from: {sender}
        Subject: {subject}
        
        Provide a helpful, professional response. If they're asking for specific analysis, offer to help and mention what visualizations you can create."""
        
        llm_response = await ollama_llm.generate_response(email_content, context)
        
        result = {
            "success": True,
            "generated_response": llm_response,
            "sender": sender,
            "subject": subject,
            "email_sent": False
        }
        
        # Auto-send if requested
        if auto_send and sender == EMAIL_CONFIG['allowed_sender']:
            reply_sent = await email_handler.send_email(
                sender,
                f"Re: {subject}" if not subject.startswith("Re:") else subject,
                llm_response
            )
            result["email_sent"] = reply_sent
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating smart response: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to generate response: {str(e)}"
        }

if __name__ == "__main__":
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    SCRIPTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    logger.info("Starting Enhanced Email MCP Server...")
    logger.info(f"Email filtering: Only {EMAIL_CONFIG['allowed_sender']}")
    logger.info(f"Ollama LLM: {OLLAMA_CONFIG['base_url']} (Model: {OLLAMA_CONFIG['model']})")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Scripts directory: {SCRIPTS_DIR}")
    
    # Start the MCP server
    mcp.run()