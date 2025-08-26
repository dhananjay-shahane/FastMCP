#!/usr/bin/env python3
"""
HTTP-enabled MCP Server for remote Claude Desktop connections
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
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from fastmcp import FastMCP, Context
except ImportError:
    print("FastMCP not available. Install with: pip install fastmcp")
    exit(1)

from utils.logger import setup_logger
from utils.ollama_client import OllamaClient
from utils.auth import authenticate_client

# Setup logging
logger = setup_logger("http_mcp_server")

# Initialize FastAPI for HTTP interface
app = FastAPI(title="MCP DataViz Server HTTP Interface")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Pydantic models for HTTP requests
class ExecuteScriptRequest(BaseModel):
    script_name: str
    args: List[str] = []

class SendEmailRequest(BaseModel):
    to: str
    subject: str
    body: str
    attachments: List[str] = []

class AnalyzeDataRequest(BaseModel):
    data_description: str
    query: str

# HTTP endpoints that wrap MCP tools
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "server": "MCP DataViz Server"}

@app.get("/resources/csv/{filename}")
async def get_csv_resource(filename: str):
    """HTTP endpoint for CSV resources"""
    try:
        data_path = Path("data") / filename
        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"CSV file {filename} not found")
        
        if not data_path.suffix.lower() == '.csv':
            raise HTTPException(status_code=400, detail=f"File {filename} is not a CSV file")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {"filename": filename, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/resources/scripts/{filename}")
async def get_script_resource(filename: str):
    """HTTP endpoint for script resources"""
    try:
        script_path = Path("scripts") / filename
        if not script_path.exists():
            raise HTTPException(status_code=404, detail=f"Script {filename} not found")
        
        if not filename.endswith('.py'):
            raise HTTPException(status_code=400, detail=f"File {filename} is not a Python script")
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {"filename": filename, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/execute_script")
async def execute_script_http(request: ExecuteScriptRequest):
    """HTTP endpoint for script execution"""
    try:
        script_path = Path("scripts") / request.script_name
        if not script_path.exists():
            raise HTTPException(status_code=404, detail=f"Script {request.script_name} not found")
        
        if not request.script_name.endswith('.py'):
            raise HTTPException(status_code=400, detail=f"File {request.script_name} is not a Python script")
        
        # Execute script with subprocess
        cmd = ["python", str(script_path)] + request.args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"Script execution failed: {result.stderr}",
                "return_code": result.returncode
            }
        
        return {
            "success": True,
            "output": result.stdout,
            "script_name": request.script_name,
            "args": request.args
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Script execution timed out (5 minutes)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/send_email")
async def send_email_http(request: SendEmailRequest):
    """HTTP endpoint for sending emails"""
    try:
        # Create message
        msg = MimeMultipart()
        msg['From'] = config['email']['username']
        msg['To'] = request.to
        msg['Subject'] = request.subject
        
        # Add body
        msg.attach(MimeText(request.body, 'plain'))
        
        # Add attachments
        for attachment_path in request.attachments:
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
        server.sendmail(config['email']['username'], request.to, text)
        server.quit()
        
        return {
            "success": True,
            "message": f"Email sent successfully to {request.to}",
            "attachments_count": len(request.attachments)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/analyze_data")
async def analyze_data_http(request: AnalyzeDataRequest):
    """HTTP endpoint for data analysis with Ollama"""
    try:
        prompt = f"""
        You are a data analysis expert. Given the following data description and user query, provide detailed insights:
        
        Data Description: {request.data_description}
        User Query: {request.query}
        
        Please provide:
        1. Direct answer to the query if possible
        2. Relevant insights from the data
        3. Recommendations for visualization if applicable
        4. Suggested Python code for analysis if needed
        """
        
        response = await ollama_client.generate(prompt)
        
        return {
            "success": True,
            "analysis": response,
            "query": request.query
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/list_resources")
async def list_resources_http():
    """HTTP endpoint to list available resources"""
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
        
        return {
            "success": True,
            "csv_files": csv_files,
            "script_files": script_files,
            "total_csv": len(csv_files),
            "total_scripts": len(script_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/system_status")
async def system_status_http():
    """HTTP endpoint for system status"""
    try:
        # Check Ollama connection
        ollama_status = await ollama_client.health_check()
        
        # Check directories
        data_exists = os.path.exists("data")
        scripts_exists = os.path.exists("scripts")
        
        # Check email configuration
        email_configured = bool(config['email']['username'] and config['email']['password'])
        
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
        raise HTTPException(status_code=500, detail=str(e))

# MCP endpoint for Claude Desktop
@app.get("/mcp")
async def mcp_endpoint():
    """MCP endpoint information"""
    return {
        "server": "DataVizServer",
        "version": "1.0.0",
        "protocol": "MCP over HTTP",
        "endpoints": {
            "resources": [
                "/resources/csv/{filename}",
                "/resources/scripts/{filename}"
            ],
            "tools": [
                "/tools/execute_script",
                "/tools/send_email", 
                "/tools/analyze_data",
                "/tools/list_resources",
                "/tools/system_status"
            ]
        }
    }

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    logger.info("Starting HTTP MCP DataViz Server...")
    logger.info(f"Data directory: {os.path.abspath('data')}")
    logger.info(f"Scripts directory: {os.path.abspath('scripts')}")
    
    # Start HTTP server
    uvicorn.run(app, host="0.0.0.0", port=8001)