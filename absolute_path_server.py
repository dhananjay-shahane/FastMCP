#!/usr/bin/env python3
"""
Standalone MCP Server with Absolute Paths - Fixed for Claude Desktop
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

# Get absolute paths based on script location
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / "data"
SCRIPTS_DIR = SCRIPT_DIR / "scripts"
LOGS_DIR = SCRIPT_DIR / "logs"
OUTPUT_DIR = SCRIPT_DIR / "output"

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
        data_path = DATA_DIR / filename
        if not data_path.exists():
            raise FileNotFoundError(f"CSV file {filename} not found in {DATA_DIR}")
        
        if not data_path.suffix.lower() == '.csv':
            raise ValueError(f"File {filename} is not a CSV file")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"Successfully read CSV resource: {filename} from {data_path}")
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
        
        logger.info(f"Successfully read script resource: {filename} from {script_path}")
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
        
        # Execute script with subprocess using absolute path
        cmd = ["python", str(script_path)] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(SCRIPT_DIR)  # Set working directory to script directory
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
        
        # List CSV files using absolute path
        if DATA_DIR.exists():
            csv_files = [f.name for f in DATA_DIR.glob("*.csv")]
        
        # List Python scripts using absolute path
        if SCRIPTS_DIR.exists():
            script_files = [f.name for f in SCRIPTS_DIR.glob("*.py")]
        
        logger.info(f"Found {len(csv_files)} CSV files and {len(script_files)} scripts")
        logger.info(f"Data directory: {DATA_DIR}")
        logger.info(f"Scripts directory: {SCRIPTS_DIR}")
        
        return {
            "success": True,
            "csv_files": csv_files,
            "script_files": script_files,
            "total_csv": len(csv_files),
            "total_scripts": len(script_files),
            "data_directory": str(DATA_DIR),
            "scripts_directory": str(SCRIPTS_DIR)
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
        # Check directories using absolute paths
        data_exists = DATA_DIR.exists()
        scripts_exists = SCRIPTS_DIR.exists()
        
        logger.info(f"System status check - Data: {data_exists}, Scripts: {scripts_exists}")
        return {
            "success": True,
            "data_directory_exists": data_exists,
            "scripts_directory_exists": scripts_exists,
            "server_name": "DataVizServer",
            "config_loaded": True,
            "script_directory": str(SCRIPT_DIR),
            "data_directory": str(DATA_DIR),
            "scripts_directory": str(SCRIPTS_DIR)
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to get system status: {str(e)}"
        }

if __name__ == "__main__":
    # Ensure directories exist using absolute paths
    DATA_DIR.mkdir(exist_ok=True)
    SCRIPTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    logger.info("Starting MCP DataViz Server with absolute paths...")
    logger.info(f"Script directory: {SCRIPT_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Scripts directory: {SCRIPTS_DIR}")
    
    # Start the MCP server (runs over stdio)
    mcp.run()