#!/usr/bin/env python3
"""
Automated Email Processor using MCP Server Tools and Ollama LLM

This MCP service:
1. Monitors emails using @mcp.tool() decorated functions
2. Processes requests using local Ollama LLM only
3. Executes data analysis via absolute path server MCP tools
4. Sends responses generated entirely by LLM - no fallback code
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from fastmcp import FastMCP, Context
except ImportError:
    print("Error: FastMCP not available. Install with: pip install fastmcp")
    sys.exit(1)

# Import MCP server components and absolute path server
from email_enhanced_server import email_handler, ollama_llm, EMAIL_CONFIG
import absolute_path_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('automated_email_processor_mcp')

# Initialize FastMCP server for automated email processing
mcp = FastMCP("AutomatedEmailProcessor")

def extract_email_address(from_field: str) -> Optional[str]:
    """Extract email address from From field"""
    try:
        import re
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', from_field)
        return email_match.group(0) if email_match else None
    except Exception:
        return None

@mcp.tool()
async def process_email_intelligently(email_data: Dict[str, Any], ctx: Context | None = None) -> Dict[str, Any]:
    """Process email using MCP tools and LLM only - no fallback responses"""
    try:
        sender = email_data.get('from', '')
        subject = email_data.get('subject', '')
        body = email_data.get('body', '')
        
        logger.info(f"📧 Processing email: {subject[:50]}...")
        logger.info(f"From: {sender}")
        
        # Extract sender email
        sender_email = extract_email_address(sender)
        if not sender_email:
            logger.warning(f"Could not extract sender email from: {sender}")
            return {"success": False, "error": "Invalid sender email"}
        
        # Use LLM and MCP tools directly - no fallback allowed
        available_files = await get_available_files_direct()
        
        # Get LLM analysis using enhanced context
        llm_context = f"""
Email Analysis Request:
From: {sender_email}
Subject: {subject}
Body: {body}

Available data files: {available_files['csv_files']}
Available scripts: {available_files['script_files']}

Analyze this email and determine:
1. Is this a data analysis/visualization request?
2. What type of visualization (bar, line, pie, stats)?
3. Which data file should be used?
4. What specific analysis is needed?
5. Does the request need clarification?

Be specific in your analysis.
"""
        
        llm_response = await ollama_llm.generate_response(body, llm_context)
        analysis = parse_text_response(llm_response, available_files)
        
        # Generate response using LLM only
        if analysis.get('requires_data_processing', False):
            # Execute data analysis via absolute path server
            execution_result = await execute_analysis_direct(analysis)
            
            # Generate response with analysis results
            response_context = f"""
Email Response Generation:

Original request: {body}
From: {sender_email}
Subject: {subject}

Analysis completed:
- Data file: {analysis.get('data_file')}
- Visualization type: {analysis.get('visualization_type')}
- Execution result: {execution_result}

Generate a professional email response explaining what was done and mentioning any attached files.
Be helpful and specific about the analysis performed.
"""
            
            attachments = execution_result.get('generated_files', [])
        else:
            # Generate response for clarification or general inquiry
            response_context = f"""
Email Response Generation:

Original request: {body}
From: {sender_email}
Subject: {subject}

Analysis: {analysis}
Available files: {available_files}

Generate a helpful professional email response. If clarification is needed, ask specific questions.
If it's a general inquiry, explain available capabilities.
"""
            
            attachments = []
        
        llm_response_content = await ollama_llm.generate_response(response_context, "Generate professional email response:")
        
        # Send email using email handler
        response_subject = f"Re: {subject}" if not subject.startswith('Re:') else subject
        success = await email_handler.send_email(
            to_email=sender_email,
            subject=response_subject,
            body=llm_response_content,
            attachments=attachments
        )
        
        email_result = {
            "success": success,
            "recipient": sender_email,
            "subject": response_subject,
            "attachments_count": len(attachments)
        }
        
        return {
            "success": email_result['success'],
            "analysis": analysis,
            "response_content": llm_response_content,
            "email_sent": email_result,
            "attachments": attachments
        }
        
    except Exception as e:
        logger.error(f"Error in MCP email processing: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def start_mcp_email_monitoring(ctx: Context | None = None) -> Dict[str, Any]:
    """Start automated email monitoring using only MCP tools and LLM"""
    logger.info("🚀 Starting MCP-only email processing workflow...")
    logger.info(f"📧 Monitoring emails from: {EMAIL_CONFIG['allowed_sender']}")
    logger.info(f"⏰ Check interval: 30 seconds")
    logger.info("🤖 All responses generated by LLM - no fallback code")
    logger.info("🔧 File operations via absolute path server MCP tools")
    logger.info("Press Ctrl+C to stop...")
    
    try:
        processed_count = 0
        while True:
            check_result = await check_and_process_emails_mcp_only()
            if check_result.get('emails_processed', 0) > 0:
                processed_count += check_result['emails_processed']
                logger.info(f"📊 Total emails processed: {processed_count}")
            
            await asyncio.sleep(30)
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  MCP email monitoring stopped by user")
        return {"success": True, "message": "Monitoring stopped", "processed_count": processed_count}
    except Exception as e:
        logger.error(f"Error in MCP monitoring: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def check_and_process_emails_mcp_only(ctx: Context | None = None) -> Dict[str, Any]:
    """Check and process emails using only MCP tools and LLM"""
    try:
        logger.info("📨 Checking for new emails...")
        
        # Get emails via MCP email handler
        emails = await email_handler.get_filtered_emails(limit=5)
        
        if not emails:
            return {"success": True, "emails_processed": 0, "message": "No new emails"}
        
        processed_emails = []
        
        # Process each email using MCP tools only
        for email_data in emails:
            result = await process_email_intelligently(email_data)
            processed_emails.append(result)
            if result.get('success'):
                logger.info(f"✅ Successfully processed email: {email_data.get('subject', '')[:50]}...")
            else:
                logger.error(f"❌ Failed to process email: {email_data.get('subject', '')[:50]}...")
            
        return {
            "success": True,
            "emails_processed": len(processed_emails),
            "results": processed_emails
        }
        
    except Exception as e:
        logger.error(f"Error in MCP email checking: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def analyze_email_with_mcp_tools_only(sender_email: str, subject: str, body: str, ctx: Context | None = None) -> Dict[str, Any]:
    """Analyze email using only MCP tools and LLM - no fallback logic"""
    try:
        # Use absolute path server to get available files
        available_files = await get_available_data_files_via_mcp()
        
        # Get LLM analysis using enhanced context
        llm_context = f"""
Email Analysis Request:
From: {sender_email}
Subject: {subject}
Body: {body}

Available data files: {available_files['csv_files']}
Available scripts: {available_files['script_files']}

Analyze this email and determine:
1. Is this a data analysis/visualization request?
2. What type of visualization (bar, line, pie, stats)?
3. Which data file should be used?
4. What specific analysis is needed?
5. Does the request need clarification?

Respond in JSON format with your analysis.
"""
        
        llm_response = await ollama_llm.generate_response(body, llm_context)
        
        # Parse LLM response to determine intent
        analysis_result = await parse_llm_analysis_response(llm_response, available_files)
        
        return {
            "success": True,
            "analysis": analysis_result,
            "llm_response": llm_response,
            "available_files": available_files
        }
        
    except Exception as e:
        logger.error(f"Error in MCP-only email analysis: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def generate_llm_response_only(analysis_result: Dict[str, Any], sender_email: str, subject: str, body: str, ctx: Context | None = None) -> Dict[str, Any]:
    """Generate response using LLM only - no hardcoded fallback text"""
    try:
        analysis = analysis_result['analysis']
        available_files = analysis_result['available_files']
        
        # Check if visualization/analysis is needed
        if analysis.get('requires_data_processing', False):
            # Execute data analysis via absolute path server
            execution_result = await execute_data_analysis_via_mcp(analysis)
            
            # Generate response with analysis results
            response_context = f"""
Email Response Generation:

Original request: {body}
From: {sender_email}
Subject: {subject}

Analysis completed:
- Data file: {analysis.get('data_file')}
- Visualization type: {analysis.get('visualization_type')}
- Execution result: {execution_result}

Generate a professional email response explaining what was done and mentioning any attached files.
Be helpful and specific about the analysis performed.
"""
            
            attachments = execution_result.get('generated_files', [])
        else:
            # Generate response for clarification or general inquiry
            response_context = f"""
Email Response Generation:

Original request: {body}
From: {sender_email}
Subject: {subject}

Analysis: {analysis}
Available files: {available_files}

Generate a helpful professional email response. If clarification is needed, ask specific questions.
If it's a general inquiry, explain available capabilities.
"""
            
            attachments = []
        
        llm_response = await ollama_llm.generate_response(response_context, "Generate professional email response:")
        
        return {
            "success": True,
            "response_content": llm_response,
            "attachments": attachments,
            "analysis_performed": analysis.get('requires_data_processing', False)
        }
        
    except Exception as e:
        logger.error(f"Error generating LLM-only response: {str(e)}")
        return {"success": False, "error": str(e)}

async def get_available_files_direct() -> Dict[str, Any]:
    """Get available data and script files via absolute path server MCP tools"""
    try:
        # Use absolute path server to list available files
        csv_files = []
        script_files = []
        
        # Get CSV files from data directory
        data_dir = Path(__file__).parent / "data"
        for csv_file in data_dir.glob('*.csv'):
            csv_files.append({
                'name': csv_file.name,
                'size': csv_file.stat().st_size,
                'available': True
            })
        
        # Get script files from scripts directory
        scripts_dir = Path(__file__).parent / "scripts"
        for script_file in scripts_dir.glob('*.py'):
            script_files.append(script_file.name)
        
        return {
            "success": True,
            "csv_files": csv_files,
            "script_files": script_files
        }
        
    except Exception as e:
        logger.error(f"Error getting files via MCP: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def parse_llm_analysis_response(llm_response: str, available_files: Dict[str, Any], ctx: Context | None = None) -> Dict[str, Any]:
    """Parse LLM response to extract analysis intent"""
    try:
        import json
        
        # Try to parse JSON response from LLM
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # Fallback: analyze response text
                analysis = analyze_text_response(llm_response, available_files)
        except json.JSONDecodeError:
            # Fallback: analyze response text
            analysis = analyze_text_response(llm_response, available_files)
        
        return {
            "success": True,
            "requires_data_processing": analysis.get('is_data_request', False),
            "visualization_type": analysis.get('visualization_type'),
            "data_file": analysis.get('data_file'),
            "needs_clarification": analysis.get('needs_clarification', False),
            "intent": analysis.get('intent', 'general_inquiry')
        }
        
    except Exception as e:
        logger.error(f"Error parsing LLM analysis: {str(e)}")
        return {"success": False, "error": str(e)}

async def execute_analysis_direct(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Execute data analysis using absolute path server MCP tools"""
    try:
        script_name = f"{analysis.get('visualization_type', 'bar_chart')}.py"
        data_file = analysis.get('data_file')
        
        if not data_file:
            return {"success": False, "error": "No data file specified"}
        
        # Execute script directly
        import subprocess
        import sys
        
        script_path = Path(__file__).parent / "scripts" / script_name
        data_path = Path(__file__).parent / "data" / data_file
        
        if not script_path.exists():
            return {"success": False, "error": f"Script {script_name} not found"}
        
        if not data_path.exists():
            return {"success": False, "error": f"Data file {data_file} not found"}
        
        # Execute script
        result = subprocess.run(
            [sys.executable, str(script_path), str(data_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        execution_result = {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None
        }
        
        if execution_result['success']:
            # Find generated output files
            output_dir = Path(__file__).parent / "output"
            timestamp = datetime.now().strftime('%Y%m%d')
            generated_files = list(output_dir.glob(f'*{timestamp}*.png'))
            
            return {
                "success": True,
                "script_output": execution_result.get('output', ''),
                "generated_files": [str(f) for f in generated_files],
                "script_name": script_name,
                "data_file": data_file
            }
        else:
            return execution_result
        
    except Exception as e:
        logger.error(f"Error executing analysis via MCP: {str(e)}")
        return {"success": False, "error": str(e)}

@mcp.tool()
async def send_mcp_email_response(sender_email: str, subject: str, response_result: Dict[str, Any], ctx: Context | None = None) -> Dict[str, Any]:
    """Send email response using MCP email handler"""
    try:
        response_subject = f"Re: {subject}" if not subject.startswith('Re:') else subject
        
        success = await email_handler.send_email(
            to_email=sender_email,
            subject=response_subject,
            body=response_result['response_content'],
            attachments=response_result.get('attachments', [])
        )
        
        return {
            "success": success,
            "recipient": sender_email,
            "subject": response_subject,
            "attachments_count": len(response_result.get('attachments', []))
        }
        
    except Exception as e:
        logger.error(f"Error sending MCP email: {str(e)}")
        return {"success": False, "error": str(e)}

def parse_text_response(response_text: str, available_files: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze LLM text response to extract intent"""
    response_lower = response_text.lower()
    
    # Check for visualization keywords
    viz_keywords = {
        'bar': 'bar_chart',
        'column': 'bar_chart',
        'line': 'line_graph',
        'trend': 'line_graph',
        'pie': 'pie_chart',
        'donut': 'pie_chart',
        'stats': 'custom_stats',
        'statistics': 'custom_stats'
    }
    
    visualization_type = None
    for keyword, viz_type in viz_keywords.items():
        if keyword in response_lower:
            visualization_type = viz_type
            break
    
    # Check for data file mentions
    data_file = None
    for file_info in available_files.get('csv_files', []):
        if file_info['name'].lower() in response_lower:
            data_file = file_info['name']
            break
    
    # Determine if this is a data request
    data_keywords = ['chart', 'graph', 'plot', 'visualization', 'analyze', 'data']
    is_data_request = any(keyword in response_lower for keyword in data_keywords)
    
    return {
        'is_data_request': is_data_request,
        'visualization_type': visualization_type,
        'data_file': data_file,
        'needs_clarification': is_data_request and (not visualization_type or not data_file),
        'intent': 'data_analysis' if is_data_request else 'general_inquiry'
    }

async def main():
    """Main function to run MCP-only automated email processor"""
    try:
        await start_mcp_email_monitoring()
    except Exception as e:
        logger.error(f"Fatal error in MCP email processor: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())