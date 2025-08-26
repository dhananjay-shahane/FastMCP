#!/usr/bin/env python3
"""
Email Utilities for MCP Server
Handles email sending, receiving, and parsing for automation workflows
"""

import smtplib
import imaplib
import email
import os
import sys
from email.mime.text import MIMEText as MimeText
from email.mime.multipart import MIMEMultipart as MimeMultipart
from email.mime.image import MIMEImage as MimeImage
from email.mime.application import MIMEApplication as MimeApplication
from datetime import datetime, timedelta
import re
import json
from typing import List, Dict, Any, Optional

class EmailHandler:
    """Handle email operations for MCP server"""
    
    def __init__(self, 
                 smtp_server: str = None,
                 smtp_port: int = None,
                 imap_server: str = None,
                 imap_port: int = None,
                 username: str = None,
                 password: str = None):
        """
        Initialize email handler with configuration
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            imap_server: IMAP server address
            imap_port: IMAP server port
            username: Email username
            password: Email password
        """
        # Load from environment variables with defaults
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.imap_server = imap_server or os.getenv('IMAP_SERVER', 'imap.gmail.com')
        self.imap_port = imap_port or int(os.getenv('IMAP_PORT', '993'))
        self.username = username or os.getenv('EMAIL_USERNAME', '')
        self.password = password or os.getenv('EMAIL_PASSWORD', '')
        
        if not self.username or not self.password:
            print("Warning: Email credentials not configured")

    def send_email(self, 
                   to: str, 
                   subject: str, 
                   body: str, 
                   attachments: List[str] = None,
                   html_body: str = None) -> Dict[str, Any]:
        """
        Send email with optional attachments
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Plain text body
            attachments: List of file paths to attach
            html_body: Optional HTML body
            
        Returns:
            Dictionary with success status and details
        """
        try:
            if attachments is None:
                attachments = []
            
            # Create message
            msg = MimeMultipart('alternative')
            msg['From'] = self.username
            msg['To'] = to
            msg['Subject'] = subject
            from email import utils as email_utils
        msg['Date'] = email_utils.formatdate(localtime=True)
            
            # Add plain text body
            text_part = MimeText(body, 'plain', 'utf-8')
            msg.attach(text_part)
            
            # Add HTML body if provided
            if html_body:
                html_part = MimeText(html_body, 'html', 'utf-8')
                msg.attach(html_part)
            
            # Add attachments
            for attachment_path in attachments:
                if os.path.exists(attachment_path):
                    self._add_attachment(msg, attachment_path)
                else:
                    print(f"Warning: Attachment not found: {attachment_path}")
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                text = msg.as_string()
                server.sendmail(self.username, to, text)
            
            return {
                'success': True,
                'message': f'Email sent successfully to {to}',
                'timestamp': datetime.now().isoformat(),
                'attachments_count': len(attachments)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to send email: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

    def _add_attachment(self, msg: MimeMultipart, file_path: str):
        """Add file attachment to email message"""
        try:
            filename = os.path.basename(file_path)
            
            # Handle different file types
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Image attachment
                with open(file_path, 'rb') as f:
                    img_data = f.read()
                image = MimeImage(img_data)
                image.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                msg.attach(image)
                
            elif file_path.lower().endswith(('.txt', '.csv', '.json', '.log')):
                # Text file attachment
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                text_attachment = MimeText(content, 'plain', 'utf-8')
                text_attachment.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                msg.attach(text_attachment)
                
            else:
                # Binary file attachment
                with open(file_path, 'rb') as f:
                    attachment_data = f.read()
                application = MimeApplication(attachment_data)
                application.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                msg.attach(application)
                
        except Exception as e:
            print(f"Error adding attachment {file_path}: {str(e)}")

    def receive_emails(self, 
                       folder: str = 'INBOX', 
                       limit: int = 10,
                       since_date: datetime = None,
                       unread_only: bool = True) -> List[Dict[str, Any]]:
        """
        Receive and parse emails from IMAP server
        
        Args:
            folder: Email folder to check
            limit: Maximum number of emails to retrieve
            since_date: Only get emails since this date
            unread_only: Only get unread emails
            
        Returns:
            List of parsed email dictionaries
        """
        try:
            emails = []
            
            # Connect to IMAP server
            with imaplib.IMAP4_SSL(self.imap_server, self.imap_port) as imap:
                imap.login(self.username, self.password)
                imap.select(folder)
                
                # Build search criteria
                search_criteria = []
                if unread_only:
                    search_criteria.append('UNSEEN')
                if since_date:
                    date_str = since_date.strftime('%d-%b-%Y')
                    search_criteria.append(f'SINCE {date_str}')
                
                search_string = ' '.join(search_criteria) if search_criteria else 'ALL'
                
                # Search for emails
                status, messages = imap.search(None, search_string)
                if status != 'OK':
                    return []
                
                # Get email IDs
                email_ids = messages[0].split()
                
                # Limit number of emails
                email_ids = email_ids[-limit:] if len(email_ids) > limit else email_ids
                
                # Process each email
                for email_id in reversed(email_ids):  # Most recent first
                    try:
                        email_data = self._parse_email(imap, email_id)
                        if email_data:
                            emails.append(email_data)
                    except Exception as e:
                        print(f"Error parsing email {email_id}: {str(e)}")
                        continue
            
            return emails
            
        except Exception as e:
            print(f"Error receiving emails: {str(e)}")
            return []

    def _parse_email(self, imap: imaplib.IMAP4_SSL, email_id: bytes) -> Optional[Dict[str, Any]]:
        """Parse individual email message"""
        try:
            # Fetch email
            status, msg_data = imap.fetch(email_id, '(RFC822)')
            if status != 'OK':
                return None
            
            # Parse email message
            raw_email = msg_data[0][1]
            email_message = email.message_from_bytes(raw_email)
            
            # Extract basic information
            parsed_email = {
                'id': email_id.decode(),
                'from': email_message.get('From', ''),
                'to': email_message.get('To', ''),
                'subject': email_message.get('Subject', ''),
                'date': email_message.get('Date', ''),
                'timestamp': datetime.now().isoformat(),
                'body': '',
                'html_body': '',
                'attachments': []
            }
            
            # Extract body content
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get('Content-Disposition', ''))
                    
                    if content_type == 'text/plain' and 'attachment' not in content_disposition:
                        body = part.get_payload(decode=True)
                        if body:
                            parsed_email['body'] = body.decode('utf-8', errors='ignore')
                    
                    elif content_type == 'text/html' and 'attachment' not in content_disposition:
                        html_body = part.get_payload(decode=True)
                        if html_body:
                            parsed_email['html_body'] = html_body.decode('utf-8', errors='ignore')
                    
                    elif 'attachment' in content_disposition:
                        filename = part.get_filename()
                        if filename:
                            parsed_email['attachments'].append(filename)
            else:
                # Simple message
                content_type = email_message.get_content_type()
                body = email_message.get_payload(decode=True)
                if body:
                    if content_type == 'text/html':
                        parsed_email['html_body'] = body.decode('utf-8', errors='ignore')
                    else:
                        parsed_email['body'] = body.decode('utf-8', errors='ignore')
            
            return parsed_email
            
        except Exception as e:
            print(f"Error parsing email: {str(e)}")
            return None

    def parse_analysis_request(self, email_body: str) -> Dict[str, Any]:
        """
        Parse email body for data analysis requests
        
        Args:
            email_body: Email body text
            
        Returns:
            Dictionary with parsed request information
        """
        try:
            request_info = {
                'is_analysis_request': False,
                'request_type': 'unknown',
                'data_file': None,
                'chart_type': None,
                'columns': [],
                'parameters': {}
            }
            
            body_lower = email_body.lower()
            
            # Check if it's an analysis request
            analysis_keywords = [
                'analyze', 'analysis', 'chart', 'graph', 'plot', 'visualization', 
                'statistics', 'stats', 'report', 'dashboard', 'data'
            ]
            
            if any(keyword in body_lower for keyword in analysis_keywords):
                request_info['is_analysis_request'] = True
            
            # Determine chart type
            chart_patterns = {
                'bar_chart': ['bar chart', 'bar graph', 'column chart'],
                'line_graph': ['line chart', 'line graph', 'trend', 'time series'],
                'pie_chart': ['pie chart', 'pie graph', 'distribution'],
                'custom_stats': ['statistics', 'stats', 'summary', 'analysis']
            }
            
            for chart_type, patterns in chart_patterns.items():
                if any(pattern in body_lower for pattern in patterns):
                    request_info['chart_type'] = chart_type
                    request_info['request_type'] = chart_type
                    break
            
            # Extract data file references
            file_patterns = [
                r'data[/\\]([a-zA-Z0-9_\-\.]+\.csv)',
                r'file[:\s]+([a-zA-Z0-9_\-\.]+\.csv)',
                r'([a-zA-Z0-9_\-\.]+\.csv)'
            ]
            
            for pattern in file_patterns:
                matches = re.findall(pattern, email_body, re.IGNORECASE)
                if matches:
                    request_info['data_file'] = matches[0]
                    break
            
            # Extract column references
            column_patterns = [
                r'column[:\s]+([a-zA-Z0-9_\-]+)',
                r'field[:\s]+([a-zA-Z0-9_\-]+)',
                r'x[:\s]*axis[:\s]+([a-zA-Z0-9_\-]+)',
                r'y[:\s]*axis[:\s]+([a-zA-Z0-9_\-]+)'
            ]
            
            for pattern in column_patterns:
                matches = re.findall(pattern, email_body, re.IGNORECASE)
                request_info['columns'].extend(matches)
            
            # Extract parameters
            param_patterns = {
                'title': r'title[:\s]+([^\n\r]+)',
                'x_column': r'x[:\s]*(?:axis|column)[:\s]+([a-zA-Z0-9_\-]+)',
                'y_column': r'y[:\s]*(?:axis|column)[:\s]+([a-zA-Z0-9_\-]+)',
                'group_by': r'group[:\s]*by[:\s]+([a-zA-Z0-9_\-]+)'
            }
            
            for param, pattern in param_patterns.items():
                match = re.search(pattern, email_body, re.IGNORECASE)
                if match:
                    request_info['parameters'][param] = match.group(1).strip()
            
            return request_info
            
        except Exception as e:
            print(f"Error parsing analysis request: {str(e)}")
            return {
                'is_analysis_request': False,
                'request_type': 'error',
                'error': str(e)
            }

    def create_response_email(self, 
                             original_email: Dict[str, Any], 
                             analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Create response email based on analysis results
        
        Args:
            original_email: Original email data
            analysis_result: Results from analysis execution
            
        Returns:
            Dictionary with response email details
        """
        try:
            # Extract sender email from 'From' field
            from_field = original_email.get('from', '')
            # Simple email extraction (could be more sophisticated)
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', from_field)
            reply_to = email_match.group(0) if email_match else ''
            
            # Create subject
            original_subject = original_email.get('subject', '')
            if not original_subject.lower().startswith('re:'):
                reply_subject = f"Re: {original_subject}"
            else:
                reply_subject = original_subject
            
            # Create response body
            if analysis_result.get('success', False):
                body = self._create_success_response(analysis_result)
            else:
                body = self._create_error_response(analysis_result)
            
            # Add greeting and signature
            full_body = f"""Hello,

Thank you for your data analysis request.

{body}

If you have any questions or need further analysis, please don't hesitate to ask.

Best regards,
MCP Data Analysis Server
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            
            return {
                'to': reply_to,
                'subject': reply_subject,
                'body': full_body,
                'attachments': analysis_result.get('attachments', [])
            }
            
        except Exception as e:
            return {
                'to': '',
                'subject': 'Error in Analysis',
                'body': f'An error occurred while creating the response: {str(e)}',
                'attachments': []
            }

    def _create_success_response(self, result: Dict[str, Any]) -> str:
        """Create success response body"""
        body = "Your analysis has been completed successfully!\n\n"
        
        if 'output' in result:
            body += f"Results:\n{result['output']}\n\n"
        
        if 'analysis' in result:
            body += f"Analysis:\n{result['analysis']}\n\n"
        
        if 'script_name' in result:
            body += f"Generated using: {result['script_name']}\n"
        
        if 'attachments' in result and result['attachments']:
            body += f"Attached files: {', '.join(os.path.basename(f) for f in result['attachments'])}\n"
        
        return body

    def _create_error_response(self, result: Dict[str, Any]) -> str:
        """Create error response body"""
        body = "I encountered an error while processing your request:\n\n"
        body += f"Error: {result.get('error', 'Unknown error')}\n\n"
        body += "Please check your request and try again. Make sure:\n"
        body += "- The CSV file name is correct\n"
        body += "- The column names exist in the data\n"
        body += "- The request format is clear\n\n"
        body += "Example request: 'Create a bar chart from sales.csv showing sales_amount by category'\n"
        
        return body

def main():
    """Test email utilities"""
    try:
        # Initialize email handler
        handler = EmailHandler()
        
        if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == 'test_send':
                # Test sending email
                to_email = input("Enter recipient email: ")
                result = handler.send_email(
                    to=to_email,
                    subject="MCP Server Test Email",
                    body="This is a test email from the MCP Data Analysis Server."
                )
                print(json.dumps(result, indent=2))
                
            elif command == 'check_inbox':
                # Test receiving emails
                emails = handler.receive_emails(limit=5, unread_only=False)
                print(f"Found {len(emails)} emails:")
                for email_data in emails:
                    print(f"- From: {email_data['from']}")
                    print(f"  Subject: {email_data['subject']}")
                    print(f"  Date: {email_data['date']}")
                    
                    # Parse for analysis requests
                    request_info = handler.parse_analysis_request(email_data['body'])
                    if request_info['is_analysis_request']:
                        print(f"  Analysis Request: {request_info['request_type']}")
                        if request_info['data_file']:
                            print(f"  Data File: {request_info['data_file']}")
                    print()
            
            elif command == 'parse_request':
                # Test request parsing
                test_body = """
                Hi, can you create a bar chart from sales.csv showing sales_amount by category?
                Title: Sales Analysis
                Please send the results as soon as possible.
                """
                result = handler.parse_analysis_request(test_body)
                print(json.dumps(result, indent=2))
                
        else:
            print("Email utilities test script")
            print("Usage: python email_utils.py [test_send|check_inbox|parse_request]")
            
    except Exception as e:
        print(f"Error in email utilities test: {str(e)}")

if __name__ == "__main__":
    main()
