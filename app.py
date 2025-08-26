#!/usr/bin/env python3
"""
Flask Web Application for MCP Server Dashboard
Provides monitoring, file uploads, and status interface
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import requests
import pandas as pd

from utils.logger import setup_logger
from utils.auth import require_auth

# Setup logging
logger = setup_logger("web_app")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev_secret_key_change_in_production')

# Configuration
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv', 'txt'}
MCP_SERVER_URL = "http://localhost:8000"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_mcp_status():
    """Check if MCP server is running"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_file_stats():
    """Get statistics about available files"""
    stats = {
        'csv_files': 0,
        'script_files': 0,
        'total_size': 0,
        'last_modified': None
    }
    
    try:
        # CSV files
        data_dir = Path(UPLOAD_FOLDER)
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            stats['csv_files'] = len(csv_files)
            
            for file in csv_files:
                stats['total_size'] += file.stat().st_size
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if not stats['last_modified'] or mtime > stats['last_modified']:
                    stats['last_modified'] = mtime
        
        # Script files
        scripts_dir = Path("scripts")
        if scripts_dir.exists():
            script_files = list(scripts_dir.glob("*.py"))
            stats['script_files'] = len(script_files)
    
    except Exception as e:
        logger.error(f"Error getting file stats: {str(e)}")
    
    return stats

@app.route('/')
def dashboard():
    """Main dashboard page"""
    try:
        # Get system status
        mcp_running = get_mcp_status()
        file_stats = get_file_stats()
        
        # Get recent files
        recent_files = []
        data_dir = Path(UPLOAD_FOLDER)
        if data_dir.exists():
            csv_files = sorted(data_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
            for file in csv_files:
                stat = file.stat()
                recent_files.append({
                    'name': file.name,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
        
        return render_template('dashboard.html', 
                             mcp_running=mcp_running,
                             file_stats=file_stats,
                             recent_files=recent_files)
    
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        flash(f"Error loading dashboard: {str(e)}", 'error')
        return render_template('dashboard.html', 
                             mcp_running=False,
                             file_stats={},
                             recent_files=[])

@app.route('/upload')
def upload_page():
    """File upload page"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Create upload directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(filepath)
            
            # Validate CSV if it's a CSV file
            if filename.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(filepath)
                    flash(f'Successfully uploaded {filename} ({len(df)} rows, {len(df.columns)} columns)', 'success')
                except Exception as e:
                    flash(f'Warning: CSV file uploaded but may have formatting issues: {str(e)}', 'warning')
            else:
                flash(f'Successfully uploaded {filename}', 'success')
            
            logger.info(f"File uploaded successfully: {filename}")
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid file type. Only CSV and TXT files are allowed.', 'error')
            return redirect(request.url)
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        flash(f'Error uploading file: {str(e)}', 'error')
        return redirect(request.url)

@app.route('/api/files')
def api_list_files():
    """API endpoint to list available files"""
    try:
        files = []
        data_dir = Path(UPLOAD_FOLDER)
        if data_dir.exists():
            for file in data_dir.glob("*.csv"):
                stat = file.stat()
                files.append({
                    'name': file.name,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'type': 'csv'
                })
        
        scripts_dir = Path("scripts")
        if scripts_dir.exists():
            for file in scripts_dir.glob("*.py"):
                stat = file.stat()
                files.append({
                    'name': file.name,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'type': 'script'
                })
        
        return jsonify({
            'success': True,
            'files': files,
            'total': len(files)
        })
    
    except Exception as e:
        logger.error(f"Error listing files via API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/file/<filename>')
def api_get_file_info(filename):
    """API endpoint to get detailed file information"""
    try:
        # Check in data directory
        file_path = Path(UPLOAD_FOLDER) / filename
        if file_path.exists() and filename.lower().endswith('.csv'):
            try:
                df = pd.read_csv(file_path)
                stat = file_path.stat()
                
                return jsonify({
                    'success': True,
                    'file_info': {
                        'name': filename,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': 'csv',
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': df.columns.tolist(),
                        'preview': df.head(5).to_dict('records')
                    }
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Error reading CSV file: {str(e)}'
                }), 400
        
        # Check in scripts directory
        script_path = Path("scripts") / filename
        if script_path.exists() and filename.endswith('.py'):
            stat = script_path.stat()
            with open(script_path, 'r') as f:
                content = f.read()
            
            return jsonify({
                'success': True,
                'file_info': {
                    'name': filename,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'type': 'script',
                    'lines': len(content.split('\n')),
                    'preview': content[:500]  # First 500 characters
                }
            })
        
        return jsonify({
            'success': False,
            'error': 'File not found'
        }), 404
    
    except Exception as e:
        logger.error(f"Error getting file info for {filename}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status')
def api_system_status():
    """API endpoint for system status"""
    try:
        mcp_running = get_mcp_status()
        file_stats = get_file_stats()
        
        return jsonify({
            'success': True,
            'status': {
                'mcp_server_running': mcp_running,
                'timestamp': datetime.now().isoformat(),
                'file_stats': file_stats
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/execute-script', methods=['POST'])
def api_execute_script():
    """API endpoint to execute scripts via MCP server"""
    try:
        data = request.get_json()
        script_name = data.get('script_name')
        args = data.get('args', [])
        
        if not script_name:
            return jsonify({
                'success': False,
                'error': 'Script name is required'
            }), 400
        
        # This would ideally call the MCP server's execute_script tool
        # For now, we'll return a placeholder response
        return jsonify({
            'success': True,
            'message': f'Script execution request submitted: {script_name}',
            'args': args
        })
    
    except Exception as e:
        logger.error(f"Error executing script via API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('dashboard.html', 
                         error="Page not found",
                         mcp_running=False,
                         file_stats={},
                         recent_files=[]), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return render_template('dashboard.html', 
                         error="Internal server error",
                         mcp_running=False,
                         file_stats={},
                         recent_files=[]), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('scripts', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Starting Flask Web Application...")
    logger.info(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
