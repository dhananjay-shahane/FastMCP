#!/usr/bin/env python3
"""
Logging utilities for MCP Server
Provides centralized logging configuration and setup
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json

class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to level name
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info'):
                log_entry[key] = value
        
        return json.dumps(log_entry)

def setup_logger(name: str, 
                config: Dict[str, Any] = None) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        config: Configuration dictionary
        
    Returns:
        Configured logger instance
    """
    # Default configuration
    default_config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': None,
        'max_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
        'console': True,
        'json_format': False,
        'colored_console': True
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    
    # Load from environment variables
    log_level = os.getenv('LOG_LEVEL', default_config['level']).upper()
    log_file = os.getenv('LOG_FILE', default_config['file'])
    log_format = os.getenv('LOG_FORMAT', default_config['format'])
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    if default_config['json_format']:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(log_format)
    
    # Console handler
    if default_config['console']:
        console_handler = logging.StreamHandler(sys.stdout)
        
        if default_config['colored_console'] and not default_config['json_format']:
            console_formatter = ColoredFormatter(log_format)
            console_handler.setFormatter(console_formatter)
        else:
            console_handler.setFormatter(formatter)
        
        console_handler.setLevel(getattr(logging, log_level))
        logger.addHandler(console_handler)
    
    # File handler
    if log_file or default_config['file']:
        log_file_path = log_file or default_config['file']
        
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=default_config['max_size'],
            backupCount=default_config['backup_count']
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level))
        logger.addHandler(file_handler)
    
    # Add request ID context if available
    try:
        from flask import has_request_context, g
        if has_request_context() and hasattr(g, 'request_id'):
            logger = logging.LoggerAdapter(logger, {'request_id': g.request_id})
    except ImportError:
        pass
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create new one with default config
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)
    
    return logger

class RequestLogger:
    """Context manager for request-specific logging"""
    
    def __init__(self, logger: logging.Logger, request_id: str = None):
        """
        Initialize request logger
        
        Args:
            logger: Base logger instance
            request_id: Unique request identifier
        """
        self.logger = logger
        self.request_id = request_id or self._generate_request_id()
        self.start_time = None
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Request started", extra={'request_id': self.request_id})
        return logging.LoggerAdapter(self.logger, {'request_id': self.request_id})
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.error(
                f"Request failed after {duration:.3f}s: {exc_val}",
                extra={'request_id': self.request_id, 'duration': duration}
            )
        else:
            self.logger.info(
                f"Request completed in {duration:.3f}s",
                extra={'request_id': self.request_id, 'duration': duration}
            )

def configure_logging_from_config(config_path: str):
    """
    Configure logging from YAML config file
    
    Args:
        config_path: Path to configuration file
    """
    try:
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logging_config = config.get('logging', {})
        
        # Configure root logger
        setup_logger('root', logging_config)
        
        # Configure specific loggers
        for logger_name, logger_config in logging_config.get('loggers', {}).items():
            setup_logger(logger_name, logger_config)
            
    except Exception as e:
        print(f"Failed to configure logging from config: {str(e)}")
        # Fall back to basic configuration
        logging.basicConfig(level=logging.INFO)

def log_performance(func):
    """
    Decorator to log function performance
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.3f}s: {str(e)}")
            raise
    
    return wrapper

def log_api_call(method: str, url: str, status_code: int, duration: float, 
                request_size: int = 0, response_size: int = 0):
    """
    Log API call details
    
    Args:
        method: HTTP method
        url: Request URL
        status_code: Response status code
        duration: Request duration in seconds
        request_size: Request size in bytes
        response_size: Response size in bytes
    """
    logger = get_logger('api')
    
    log_data = {
        'method': method,
        'url': url,
        'status_code': status_code,
        'duration': duration,
        'request_size': request_size,
        'response_size': response_size
    }
    
    if status_code >= 500:
        logger.error(f"API call failed: {method} {url}", extra=log_data)
    elif status_code >= 400:
        logger.warning(f"API call client error: {method} {url}", extra=log_data)
    else:
        logger.info(f"API call: {method} {url}", extra=log_data)

def setup_system_logging():
    """Setup system-wide logging configuration"""
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Main application logger
    app_logger = setup_logger('mcp_server', {
        'level': 'INFO',
        'file': 'logs/mcp_server.log',
        'console': True,
        'colored_console': True
    })
    
    # API logger
    api_logger = setup_logger('api', {
        'level': 'INFO',
        'file': 'logs/api.log',
        'console': False
    })
    
    # Error logger
    error_logger = setup_logger('error', {
        'level': 'ERROR',
        'file': 'logs/error.log',
        'console': True
    })
    
    # Performance logger
    perf_logger = setup_logger('performance', {
        'level': 'DEBUG',
        'file': 'logs/performance.log',
        'console': False
    })
    
    return {
        'app': app_logger,
        'api': api_logger,
        'error': error_logger,
        'performance': perf_logger
    }

# Test logging setup
def test_logging():
    """Test logging configuration"""
    try:
        print("Testing logging setup...")
        
        # Setup test logger
        logger = setup_logger('test', {
            'level': 'DEBUG',
            'console': True,
            'colored_console': True
        })
        
        # Test different log levels
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
        
        # Test with extra data
        logger.info("Message with extra data", extra={'user_id': 123, 'action': 'test'})
        
        # Test request logger
        with RequestLogger(logger) as req_logger:
            req_logger.info("Inside request context")
        
        print("Logging test completed successfully")
        
    except Exception as e:
        print(f"Logging test failed: {str(e)}")

if __name__ == "__main__":
    test_logging()
