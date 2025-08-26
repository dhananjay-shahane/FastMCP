#!/usr/bin/env python3
"""
MCP Server implementation using FastMCP Python SDK
Exposes CSV data resources and Python script execution tools
"""

import os
import subprocess
import smtplib
import logging
from email.mime.text import MIMEText as MimeText
from email.mime.multipart import MIMEMultipart as MimeMultipart
from email.mime.image import MIMEImage as MimeImage
from pathlib import Path
from typing import List, Dict, Any
import yaml
import json

try:
    from fastmcp import FastMCP, Context
except ImportError:
    print("FastMCP not available. Install with: pip install fastmcp")
    exit(1)

from utils.logger import setup_logger
from utils.ollama_client import OllamaClient
from utils.auth import authenticate_client

# Setup logging
logger = setup_logger("mcp_server")

# Initialize FastMCP server
mcp = FastMCP("DataVizServer")

# Load configuration
config_path = "mcp_config.yaml"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
else:
    config = {
        'email': {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('EMAIL_USERNAME', ''),
            'password': os.getenv('EMAIL_PASSWORD', '')
        },
        'ollama': {
            'host': os.getenv('OLLAMA_HOST', 'localhost'),
            'port': int(os.getenv('OLLAMA_PORT', '11434'))
        },
        'auth': {
            'api_key': os.getenv('MCP_API_KEY', 'default_key')
        }
    }

# Initialize Ollama client
ollama_client = OllamaClient(
    host=config['ollama']['host'],
    port=config['ollama']['port']
)

# Resources - CSV files under /data folder
@mcp.resource("file://data/{filename}")
async def read_csv_resource(filename: str) -> str:
    """Read CSV files from the data directory"""
    try:
        data_path = Path("data") / filename
        if not data_path.exists():
            raise FileNotFoundError(f"CSV file {filename} not found in data directory")
        
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
        script_path = Path("scripts") / filename
        if not script_path.exists():
            raise FileNotFoundError(f"Script file {filename} not found in scripts directory")
        
        if not filename.endswith('.py'):
            raise ValueError(f"File {filename} is not a Python script")
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"Successfully read script resource: {filename}")
        return content
        
    except Exception as e:
        logger.error(f"Error reading script resource {filename}: {str(e)}")
        raise

# Tools - Script execution and email functionality
@mcp.tool()
async def execute_script(script_name: str, args: List[str] = [], ctx: Context = None) -> Dict[str, Any]:
    """Execute Python scripts dynamically and return results"""
    try:
        if args is None:
            args = []
            
        script_path = Path("scripts") / script_name
        if not script_path.exists():
            return {
                "success": False,
                "error": f"Script {script_name} not found in scripts directory"
            }
        
        if not script_name.endswith('.py'):
            return {
                "success": False,
                "error": f"File {script_name} is not a Python script"
            }
        
        # Execute script with subprocess
        cmd = ["python", str(script_path)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.getcwd()
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
        logger.error(f"Script execution timeout: {script_name}")
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
async def send_email(to: str, subject: str, body: str, attachments: List[str] = [], ctx: Context = None) -> Dict[str, Any]:
    """Send email with optional attachments"""
    try:
        if attachments is None:
            attachments = []
        
        # Create message
        msg = MimeMultipart()
        msg['From'] = config['email']['username']
        msg['To'] = to
        msg['Subject'] = subject
        
        # Add body
        msg.attach(MimeText(body, 'plain'))
        
        # Add attachments
        for attachment_path in attachments:
            if os.path.exists(attachment_path):
                if attachment_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    with open(attachment_path, 'rb') as f:
                        img_data = f.read()
                    image = MimeImage(img_data)
                    image.add_header('Content-Disposition', f'attachment; filename={os.path.basename(attachment_path)}')
                    msg.attach(image)
                else:
                    with open(attachment_path, 'rb') as f:
                        attachment_data = f.read()
                    attachment = MimeText(attachment_data.decode('utf-8', errors='ignore'))
                    attachment.add_header('Content-Disposition', f'attachment; filename={os.path.basename(attachment_path)}')
                    msg.attach(attachment)
        
        # Send email
        server = smtplib.SMTP(config['email']['smtp_server'], config['email']['smtp_port'])
        server.starttls()
        server.login(config['email']['username'], config['email']['password'])
        text = msg.as_string()
        server.sendmail(config['email']['username'], to, text)
        server.quit()
        
        logger.info(f"Successfully sent email to: {to}")
        return {
            "success": True,
            "message": f"Email sent successfully to {to}",
            "attachments_count": len(attachments)
        }
        
    except Exception as e:
        logger.error(f"Error sending email to {to}: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to send email: {str(e)}"
        }

@mcp.tool()
async def analyze_data_with_ollama(data_description: str, query: str, ctx: Context = None) -> Dict[str, Any]:
    """Use Ollama LLM to analyze data and provide insights"""
    try:
        prompt = f"""
        You are a data analysis expert. Given the following data description and user query, provide detailed insights:
        
        Data Description: {data_description}
        User Query: {query}
        
        Please provide:
        1. Direct answer to the query if possible
        2. Relevant insights from the data
        3. Recommendations for visualization if applicable
        4. Suggested Python code for analysis if needed
        """
        
        response = await ollama_client.generate(prompt)
        
        logger.info(f"Successfully analyzed data with Ollama")
        return {
            "success": True,
            "analysis": response,
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Error analyzing data with Ollama: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to analyze data: {str(e)}"
        }

@mcp.tool()
async def list_available_resources(ctx: Context = None) -> Dict[str, Any]:
    """List all available CSV files and Python scripts"""
    try:
        csv_files = []
        script_files = []
        
        # List CSV files
        data_dir = Path("data")
        if data_dir.exists():
            csv_files = [f.name for f in data_dir.glob("*.csv")]
        
        # List Python scripts
        scripts_dir = Path("scripts")
        if scripts_dir.exists():
            script_files = [f.name for f in scripts_dir.glob("*.py")]
        
        logger.info("Successfully listed available resources")
        return {
            "success": True,
            "csv_files": csv_files,
            "script_files": script_files,
            "total_csv": len(csv_files),
            "total_scripts": len(script_files)
        }
        
    except Exception as e:
        logger.error(f"Error listing resources: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to list resources: {str(e)}"
        }

@mcp.tool()
async def get_system_status(ctx: Context = None) -> Dict[str, Any]:
    """Get system status and health information"""
    try:
        # Check Ollama connection
        ollama_status = await ollama_client.health_check()
        
        # Check directories
        data_exists = os.path.exists("data")
        scripts_exists = os.path.exists("scripts")
        
        # Check email configuration
        email_configured = bool(config['email']['username'] and config['email']['password'])
        
        logger.info("Successfully retrieved system status")
        return {
            "success": True,
            "ollama_connected": ollama_status,
            "data_directory_exists": data_exists,
            "scripts_directory_exists": scripts_exists,
            "email_configured": email_configured,
            "server_name": "DataVizServer",
            "config_loaded": True
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to get system status: {str(e)}"
        }

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    logger.info("Starting MCP DataViz Server...")
    logger.info(f"Data directory: {os.path.abspath('data')}")
    logger.info(f"Scripts directory: {os.path.abspath('scripts')}")
    
    # Start the MCP server (runs over stdio)
    mcp.run()
