#!/usr/bin/env python3
"""
Authentication utilities for MCP Server
Handles API key validation and access control
"""

import os
import hashlib
import hmac
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps
from flask import request, jsonify, g
import jwt

# Setup logging
logger = logging.getLogger(__name__)

class AuthManager:
    """Manages authentication and authorization for MCP server"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize authentication manager
        
        Args:
            config: Configuration dictionary with auth settings
        """
        self.config = config or {}
        
        # Load configuration from environment or config
        self.api_key = self.config.get('api_key') or os.getenv('MCP_API_KEY', 'default_key')
        self.secret_key = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))
        self.require_auth = self.config.get('require_auth', True)
        self.allowed_origins = self.config.get('allowed_origins', ['*'])
        self.token_expiry_hours = self.config.get('token_expiry_hours', 24)
        
        # Valid API keys (in production, use database)
        self.valid_api_keys = {
            self.api_key: {
                'name': 'default',
                'permissions': ['read', 'write', 'execute'],
                'created': datetime.now().isoformat()
            }
        }
        
        # Load additional API keys from environment
        self._load_additional_keys()
        
        logger.info(f"Auth manager initialized - require_auth: {self.require_auth}")
    
    def _load_additional_keys(self):
        """Load additional API keys from environment variables"""
        try:
            # Look for MCP_API_KEYS environment variable (JSON format)
            additional_keys = os.getenv('MCP_API_KEYS')
            if additional_keys:
                import json
                keys_data = json.loads(additional_keys)
                for key, data in keys_data.items():
                    self.valid_api_keys[key] = data
                    logger.info(f"Loaded API key: {data.get('name', 'unnamed')}")
        except Exception as e:
            logger.warning(f"Failed to load additional API keys: {str(e)}")
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key and return associated metadata
        
        Args:
            api_key: API key to validate
            
        Returns:
            Key metadata if valid, None otherwise
        """
        if not api_key or not self.require_auth:
            if not self.require_auth:
                return {'name': 'public', 'permissions': ['read']}
            return None
        
        # Simple key validation
        key_info = self.valid_api_keys.get(api_key)
        if key_info:
            logger.debug(f"Valid API key used: {key_info.get('name', 'unnamed')}")
            return key_info
        
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        return None
    
    def generate_jwt_token(self, api_key_info: Dict[str, Any]) -> str:
        """
        Generate JWT token for authenticated session
        
        Args:
            api_key_info: API key metadata
            
        Returns:
            JWT token string
        """
        try:
            payload = {
                'name': api_key_info.get('name'),
                'permissions': api_key_info.get('permissions', []),
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            return token
            
        except Exception as e:
            logger.error(f"Error generating JWT token: {str(e)}")
            raise
    
    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token and return payload
        
        Args:
            token: JWT token to validate
            
        Returns:
            Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {str(e)}")
            return None
    
    def check_permission(self, user_info: Dict[str, Any], required_permission: str) -> bool:
        """
        Check if user has required permission
        
        Args:
            user_info: User information from token/API key
            required_permission: Required permission string
            
        Returns:
            True if user has permission, False otherwise
        """
        user_permissions = user_info.get('permissions', [])
        return required_permission in user_permissions or 'admin' in user_permissions
    
    def check_origin(self, origin: str) -> bool:
        """
        Check if request origin is allowed
        
        Args:
            origin: Request origin header
            
        Returns:
            True if origin is allowed, False otherwise
        """
        if '*' in self.allowed_origins:
            return True
        
        return origin in self.allowed_origins
    
    def hash_api_key(self, api_key: str) -> str:
        """
        Hash API key for secure storage
        
        Args:
            api_key: API key to hash
            
        Returns:
            Hashed API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def generate_api_key(self, name: str, permissions: List[str] = None) -> str:
        """
        Generate new API key
        
        Args:
            name: Name for the API key
            permissions: List of permissions to grant
            
        Returns:
            Generated API key
        """
        if permissions is None:
            permissions = ['read']
        
        # Generate secure random API key
        api_key = f"mcp_{secrets.token_urlsafe(32)}"
        
        # Store key info
        self.valid_api_keys[api_key] = {
            'name': name,
            'permissions': permissions,
            'created': datetime.now().isoformat()
        }
        
        logger.info(f"Generated new API key: {name}")
        return api_key

# Global auth manager instance
auth_manager = AuthManager()

def require_auth(permission: str = 'read'):
    """
    Decorator to require authentication for Flask routes
    
    Args:
        permission: Required permission level
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not auth_manager.require_auth:
                # Auth disabled, allow access
                g.user = {'name': 'public', 'permissions': ['read', 'write']}
                return f(*args, **kwargs)
            
            # Try API key authentication first
            api_key = None
            
            # Check Authorization header
            auth_header = request.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                token_or_key = auth_header[7:]
                
                # Try as JWT token first
                user_info = auth_manager.validate_jwt_token(token_or_key)
                if user_info:
                    if auth_manager.check_permission(user_info, permission):
                        g.user = user_info
                        return f(*args, **kwargs)
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'Insufficient permissions. Required: {permission}'
                        }), 403
                
                # Try as API key
                api_key = token_or_key
            
            # Check X-API-Key header
            if not api_key:
                api_key = request.headers.get('X-API-Key')
            
            # Check query parameter
            if not api_key:
                api_key = request.args.get('api_key')
            
            if api_key:
                key_info = auth_manager.validate_api_key(api_key)
                if key_info and auth_manager.check_permission(key_info, permission):
                    g.user = key_info
                    return f(*args, **kwargs)
            
            # Authentication failed
            return jsonify({
                'success': False,
                'error': 'Authentication required'
            }), 401
        
        return decorated_function
    return decorator

def authenticate_client(client_info: Dict[str, Any]) -> bool:
    """
    Authenticate MCP client connection
    
    Args:
        client_info: Client information dictionary
        
    Returns:
        True if client is authenticated, False otherwise
    """
    try:
        # Check if authentication is required
        if not auth_manager.require_auth:
            return True
        
        # Validate API key if provided
        api_key = client_info.get('api_key')
        if api_key:
            key_info = auth_manager.validate_api_key(api_key)
            return key_info is not None
        
        # Check origin if provided
        origin = client_info.get('origin')
        if origin:
            return auth_manager.check_origin(origin)
        
        return False
        
    except Exception as e:
        logger.error(f"Error authenticating client: {str(e)}")
        return False

def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Get current authenticated user from Flask g object
    
    Returns:
        Current user info or None
    """
    return getattr(g, 'user', None)

def create_auth_token(api_key: str) -> Optional[str]:
    """
    Create authentication token for API key
    
    Args:
        api_key: Valid API key
        
    Returns:
        JWT token if successful, None otherwise
    """
    try:
        key_info = auth_manager.validate_api_key(api_key)
        if key_info:
            return auth_manager.generate_jwt_token(key_info)
        return None
    except Exception as e:
        logger.error(f"Error creating auth token: {str(e)}")
        return None

def validate_cors_origin(origin: str) -> bool:
    """
    Validate CORS origin
    
    Args:
        origin: Request origin
        
    Returns:
        True if origin is valid, False otherwise
    """
    return auth_manager.check_origin(origin)

# Configuration helpers
def configure_auth(config: Dict[str, Any]):
    """
    Configure authentication with new settings
    
    Args:
        config: Configuration dictionary
    """
    global auth_manager
    auth_manager = AuthManager(config)

def get_auth_config() -> Dict[str, Any]:
    """
    Get current authentication configuration
    
    Returns:
        Auth configuration dictionary
    """
    return {
        'require_auth': auth_manager.require_auth,
        'allowed_origins': auth_manager.allowed_origins,
        'token_expiry_hours': auth_manager.token_expiry_hours,
        'api_keys_count': len(auth_manager.valid_api_keys)
    }

# Test authentication
def test_auth():
    """Test authentication functionality"""
    try:
        print("Testing authentication...")
        
        # Test API key validation
        test_key = "test_key_123"
        auth_manager.valid_api_keys[test_key] = {
            'name': 'test',
            'permissions': ['read', 'write'],
            'created': datetime.now().isoformat()
        }
        
        key_info = auth_manager.validate_api_key(test_key)
        print(f"API key validation: {'✓' if key_info else '✗'}")
        
        if key_info:
            # Test JWT token generation
            token = auth_manager.generate_jwt_token(key_info)
            print(f"JWT generation: {'✓' if token else '✗'}")
            
            # Test JWT validation
            payload = auth_manager.validate_jwt_token(token)
            print(f"JWT validation: {'✓' if payload else '✗'}")
            
            # Test permission check
            has_permission = auth_manager.check_permission(key_info, 'read')
            print(f"Permission check: {'✓' if has_permission else '✗'}")
        
        print("Authentication test completed")
        
    except Exception as e:
        print(f"Authentication test failed: {str(e)}")

if __name__ == "__main__":
    test_auth()
