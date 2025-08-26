#!/usr/bin/env python3
"""
Standalone MCP Server - No external dependencies required
"""

import os
import subprocess
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

try:
    from fastmcp import FastMCP, Context
except ImportError:
    print("Error: FastMCP not available. Install with: pip install fastmcp")
    sys.exit(1)

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_server")

# Initialize FastMCP server
mcp = FastMCP("DataVizServer")

# Basic configuration
config = {
    'email': {
        'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'username': os.getenv('EMAIL_USERNAME', ''),
        'password': os.getenv('EMAIL_PASSWORD', '')
    },
    'auth': {
        'api_key': os.getenv('MCP_API_KEY', 'default_key')
    }
}

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

# Tools - Script execution
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
        # Check directories
        data_exists = os.path.exists("data")
        scripts_exists = os.path.exists("scripts")
        
        logger.info("Successfully retrieved system status")
        return {
            "success": True,
            "data_directory_exists": data_exists,
            "scripts_directory_exists": scripts_exists,
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