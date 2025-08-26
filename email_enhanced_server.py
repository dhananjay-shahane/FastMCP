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
import email
import imaplib
import smtplib
from pathlib import Path
from typing import List, Dict, Any, Optional
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.header import decode_header
import ssl
from datetime import datetime

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

# Email and LLM Configuration
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'imap_server': 'imap.gmail.com',
    'imap_port': 993,
    'username': 'dhanushahane01@gmail.com',
    'password': 'sljo pinu ajrh padp',
    'allowed_sender': 'dhananjayshahane24@gmail.com'  # Filter emails only from this address
}

OLLAMA_CONFIG = {
    'base_url': 'http://127.0.0.1:11434',
    'model': 'llama3.2',  # Default model, can be changed
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
                await self.connect_imap()
            
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
                if status == 'OK':
                    email_message = email.message_from_bytes(msg_data[0][1])
                    
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
    
    async def send_email(self, to_email: str, subject: str, body: str):
        """Send email response"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.config['username']
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['username'], self.config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False

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
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Ollama server not available: {str(e)}")
            return False
    
    async def generate_response(self, prompt: str, context: str = ""):
        """Generate response using Ollama"""
        try:
            full_prompt = f"""You are an AI assistant helping with data analysis requests via email. 

Context: {context}

Email Content: {prompt}

Please provide a helpful response. If the email is asking for data analysis, explain what analysis you can perform with CSV data files. Be professional and concise."""

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
            logger.error(f"Error generating Ollama response: {str(e)}")
            return "Sorry, I'm currently unable to process your request."

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
async def check_filtered_emails(limit: int = 5, ctx: Context = None) -> Dict[str, Any]:
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
async def process_email_with_llm(email_content: str, sender: str, subject: str, ctx: Context = None) -> Dict[str, Any]:
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
async def send_automated_reply(to_email: str, subject: str, response_content: str, ctx: Context = None) -> Dict[str, Any]:
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
async def check_and_respond_to_emails(ctx: Context = None) -> Dict[str, Any]:
    """Complete workflow: Check emails, process with LLM, and send responses"""
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
                # Process with LLM
                llm_response = await ollama_llm.generate_response(
                    email_data['body'], 
                    f"Email from: {email_data['from']}, Subject: {email_data['subject']}"
                )
                
                # Send reply
                reply_sent = await email_handler.send_email(
                    EMAIL_CONFIG['allowed_sender'],
                    f"Re: {email_data['subject']}",
                    llm_response
                )
                
                processed_emails.append({
                    "email_id": email_data['id'],
                    "subject": email_data['subject'],
                    "reply_sent": reply_sent,
                    "response_preview": llm_response[:100] + "..."
                })
                
            except Exception as email_error:
                logger.error(f"Error processing individual email: {str(email_error)}")
                continue
        
        logger.info(f"Processed {len(processed_emails)} emails with automated responses")
        return {
            "success": True,
            "processed_count": len(processed_emails),
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
async def check_system_status(ctx: Context = None) -> Dict[str, Any]:
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

# Keep existing script execution tools
@mcp.tool()
async def execute_script(script_name: str, args: List[str] = [], ctx: Context = None) -> Dict[str, Any]:
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
async def list_available_resources(ctx: Context = None) -> Dict[str, Any]:
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

if __name__ == "__main__":
    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    SCRIPTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    logger.info("Starting Enhanced Email MCP Server...")
    logger.info(f"Email filtering: Only {EMAIL_CONFIG['allowed_sender']}")
    logger.info(f"Ollama LLM: {OLLAMA_CONFIG['base_url']}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Scripts directory: {SCRIPTS_DIR}")
    
    # Start the MCP server
    mcp.run()