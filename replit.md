# Overview

This is an MCP (Model Context Protocol) Data Analysis Server that provides automated data analysis and visualization capabilities through email-driven workflows. The system combines a Python MCP server built with FastMCP, a Flask web dashboard, n8n workflow automation, and local Ollama LLM integration to create a comprehensive data processing pipeline.

The application accepts CSV data files, generates various visualizations (bar charts, line graphs, pie charts), performs statistical analysis, and can respond to user requests via email automation. It's designed to run locally while supporting remote MCP client connections.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Core Components

**MCP Server (FastMCP)**
- Built using the official MCP Python SDK (FastMCP)
- Exposes CSV files in `/data` folder as resources via `file://data/{filename}` URIs
- Provides script execution tools for dynamic Python script running
- Implements email sending/receiving tools for workflow automation
- Supports both local and remote MCP client connections
- Integrates with local Ollama LLM for language processing

**Web Dashboard (Flask)**
- Flask-based web application providing monitoring interface
- File upload functionality with drag-and-drop support
- System status monitoring and health checks
- Dashboard for viewing available files and execution results
- Bootstrap frontend with responsive design
- File size limit of 16MB for uploads

**Visualization Scripts**
- Modular Python scripts for different chart types (bar, line, pie)
- Statistical analysis script for comprehensive data reports
- Email utilities for automated communication
- All scripts support CSV input from `/data` directory
- Generated outputs saved to `/output` directory

**Authentication & Security**
- API key-based authentication system
- JWT token support for secure sessions
- Configurable access control with permissions
- CORS support for cross-origin requests

**Email Automation (n8n)**
- Workflow automation for processing incoming emails
- Email parsing for data analysis requests
- Integration with MCP server tools via HTTP requests
- Automatic routing based on request type

## Data Flow Architecture

**Resource Management**
- CSV files stored in `/data` directory
- Python scripts organized in `/scripts` directory
- Generated outputs saved to `/output` directory
- File validation for allowed extensions (CSV, TXT)

**Execution Pipeline**
1. Email/web request triggers analysis
2. MCP server validates request and authenticates client
3. Appropriate script is selected or generated dynamically
4. Data is processed and visualizations created
5. Results are returned via email or web interface

**LLM Integration**
- Local Ollama server provides language model capabilities
- Configurable model selection and timeout settings
- Health checking for LLM availability
- Asynchronous communication for better performance

# External Dependencies

**Core Framework Dependencies**
- FastMCP: Official MCP Python SDK for server implementation
- Flask: Web application framework for dashboard
- Pandas: Data manipulation and analysis
- Matplotlib/Seaborn: Data visualization libraries
- PyYAML: Configuration file parsing

**Email Services**
- SMTP server configuration (default: Gmail)
- IMAP server for receiving emails
- Support for various email providers through environment variables

**LLM Backend**
- Ollama: Local LLM server (default port 11434)
- Configurable model selection
- HTTP-based API communication

**Workflow Automation**
- n8n: Visual workflow automation platform
- Email triggers and HTTP request nodes
- JSON-based workflow configuration

**Database & Storage**
- File system based storage for CSV data and outputs
- No external database required for core functionality
- Optional JWT secret key for session management

**Monitoring & Logging**
- Centralized logging with colored console output
- JSON structured logging option
- Health check endpoints for system monitoring
- Bootstrap and Font Awesome for UI components