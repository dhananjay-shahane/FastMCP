#!/usr/bin/env python3
"""
Ollama Client for MCP Server
Handles communication with local Ollama LLM instance
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional, List
import os
from datetime import datetime

class OllamaClient:
    """Client for interacting with Ollama LLM server"""
    
    def __init__(self, host: str = "localhost", port: int = 11434, model: str = None):
        """
        Initialize Ollama client
        
        Args:
            host: Ollama server host
            port: Ollama server port
            model: Default model to use
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.model = model or os.getenv('OLLAMA_MODEL', 'llama2')
        self.timeout = int(os.getenv('OLLAMA_TIMEOUT', '60'))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def health_check(self) -> bool:
        """
        Check if Ollama server is running and accessible
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"Ollama health check failed: {str(e)}")
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models on Ollama server
        
        Returns:
            List of model information dictionaries
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('models', [])
                    else:
                        self.logger.error(f"Failed to list models: HTTP {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return []
    
    async def generate(self, 
                      prompt: str, 
                      model: str = None,
                      system_prompt: str = None,
                      temperature: float = 0.7,
                      max_tokens: int = None,
                      stream: bool = False) -> str:
        """
        Generate text completion using Ollama
        
        Args:
            prompt: Input prompt for generation
            model: Model to use (defaults to instance model)
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            
        Returns:
            Generated text response
        """
        try:
            model_name = model or self.model
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    
                    if stream:
                        return await self._handle_streaming_response(response)
                    else:
                        return await self._handle_single_response(response)
                        
        except asyncio.TimeoutError:
            self.logger.error("Ollama request timed out")
            raise Exception("Request timed out - consider using a smaller prompt or increasing timeout")
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            raise Exception(f"Failed to generate text: {str(e)}")
    
    async def _handle_single_response(self, response: aiohttp.ClientResponse) -> str:
        """Handle non-streaming response"""
        try:
            data = await response.json()
            return data.get('response', '')
        except Exception as e:
            self.logger.error(f"Error parsing response: {str(e)}")
            raise Exception("Failed to parse response from Ollama")
    
    async def _handle_streaming_response(self, response: aiohttp.ClientResponse) -> str:
        """Handle streaming response"""
        try:
            full_response = ""
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            full_response += data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            return full_response
        except Exception as e:
            self.logger.error(f"Error handling streaming response: {str(e)}")
            raise Exception("Failed to process streaming response")
    
    async def chat(self, 
                   messages: List[Dict[str, str]], 
                   model: str = None,
                   temperature: float = 0.7,
                   stream: bool = False) -> str:
        """
        Chat completion using conversation format
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use
            temperature: Sampling temperature
            stream: Whether to stream response
            
        Returns:
            Generated response text
        """
        try:
            model_name = model or self.model
            
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": temperature
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")
                    
                    if stream:
                        return await self._handle_streaming_chat_response(response)
                    else:
                        return await self._handle_single_chat_response(response)
                        
        except Exception as e:
            self.logger.error(f"Error in chat completion: {str(e)}")
            raise Exception(f"Failed to complete chat: {str(e)}")
    
    async def _handle_single_chat_response(self, response: aiohttp.ClientResponse) -> str:
        """Handle non-streaming chat response"""
        try:
            data = await response.json()
            message = data.get('message', {})
            return message.get('content', '')
        except Exception as e:
            self.logger.error(f"Error parsing chat response: {str(e)}")
            raise Exception("Failed to parse chat response from Ollama")
    
    async def _handle_streaming_chat_response(self, response: aiohttp.ClientResponse) -> str:
        """Handle streaming chat response"""
        try:
            full_response = ""
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        message = data.get('message', {})
                        if 'content' in message:
                            full_response += message['content']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            return full_response
        except Exception as e:
            self.logger.error(f"Error handling streaming chat response: {str(e)}")
            raise Exception("Failed to process streaming chat response")
    
    async def analyze_data_request(self, request_text: str, available_data: List[str]) -> Dict[str, Any]:
        """
        Analyze a user request to determine data analysis intent
        
        Args:
            request_text: User's request text
            available_data: List of available data files
            
        Returns:
            Analysis results with recommendations
        """
        try:
            system_prompt = """You are a data analysis expert. Analyze user requests and provide structured recommendations for data visualization and analysis.

Available data files: """ + ", ".join(available_data) + """

For each request, determine:
1. Intent (chart type, analysis type)
2. Relevant data files
3. Suggested columns/parameters
4. Visualization recommendations

Respond in JSON format with: intent, data_file, chart_type, parameters, explanation"""
            
            response = await self.generate(
                prompt=request_text,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            # Try to parse JSON response
            try:
                analysis = json.loads(response)
                return {
                    'success': True,
                    'analysis': analysis,
                    'raw_response': response
                }
            except json.JSONDecodeError:
                # If not valid JSON, return as text analysis
                return {
                    'success': True,
                    'analysis': {
                        'intent': 'analysis',
                        'explanation': response
                    },
                    'raw_response': response
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing data request: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def generate_code(self, 
                           task_description: str, 
                           data_info: Dict[str, Any],
                           code_type: str = "python") -> str:
        """
        Generate code for data analysis tasks
        
        Args:
            task_description: Description of the task
            data_info: Information about the data (columns, types, etc.)
            code_type: Type of code to generate
            
        Returns:
            Generated code as string
        """
        try:
            system_prompt = f"""You are an expert {code_type} programmer specializing in data analysis and visualization.
Generate clean, well-commented code based on the task description and data information provided.

Data Information:
{json.dumps(data_info, indent=2)}

Guidelines:
- Use pandas for data manipulation
- Use matplotlib/seaborn for visualization
- Include error handling
- Add helpful comments
- Make code production-ready"""
            
            response = await self.generate(
                prompt=f"Task: {task_description}",
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=1000
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating code: {str(e)}")
            raise Exception(f"Failed to generate code: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get client status information
        
        Returns:
            Dictionary with client status
        """
        return {
            'host': self.host,
            'port': self.port,
            'base_url': self.base_url,
            'model': self.model,
            'timeout': self.timeout,
            'timestamp': datetime.now().isoformat()
        }

# Async context manager for automatic cleanup
class OllamaClientManager:
    """Context manager for Ollama client"""
    
    def __init__(self, **kwargs):
        self.client_kwargs = kwargs
        self.client = None
    
    async def __aenter__(self):
        self.client = OllamaClient(**self.client_kwargs)
        # Test connection
        if not await self.client.health_check():
            raise Exception("Failed to connect to Ollama server")
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass

# Convenience function for quick access
async def create_ollama_client(host: str = "localhost", 
                              port: int = 11434, 
                              model: str = None) -> OllamaClient:
    """
    Create and test Ollama client connection
    
    Args:
        host: Ollama server host
        port: Ollama server port
        model: Default model name
        
    Returns:
        Connected OllamaClient instance
    """
    client = OllamaClient(host=host, port=port, model=model)
    
    if not await client.health_check():
        raise Exception(f"Cannot connect to Ollama server at {host}:{port}")
    
    return client

# Test function
async def main():
    """Test Ollama client functionality"""
    try:
        client = await create_ollama_client()
        
        print("Testing Ollama connection...")
        
        # Test health check
        healthy = await client.health_check()
        print(f"Health check: {'✓' if healthy else '✗'}")
        
        if healthy:
            # Test model listing
            models = await client.list_models()
            print(f"Available models: {len(models)}")
            for model in models[:3]:  # Show first 3
                print(f"  - {model.get('name', 'Unknown')}")
            
            # Test generation
            print("\nTesting generation...")
            response = await client.generate(
                prompt="What is data analysis? Respond in one sentence.",
                temperature=0.3
            )
            print(f"Response: {response[:100]}...")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
