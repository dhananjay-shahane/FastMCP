
#!/usr/bin/env python3
"""
Test script to check email connectivity and configuration
"""

import os
import imaplib
import smtplib
from dotenv import load_dotenv

def test_email_connection():
    """Test email connection with current configuration"""
    load_dotenv()
    
    # Get email configuration
    username = os.getenv('EMAIL_USER')
    password = os.getenv('EMAIL_PASSWORD')
    imap_server = 'imap.gmail.com'
    imap_port = 993
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    
    print("Testing Email Configuration...")
    print(f"Username: {username}")
    print(f"Password: {'*' * len(password) if password else 'NOT SET'}")
    
    if not username or not password:
        print("❌ ERROR: Email credentials not found in .env file")
        print("Please add EMAIL_USER and EMAIL_PASSWORD to your .env file")
        return False
    
    # Test IMAP connection
    print(f"\n📨 Testing IMAP connection to {imap_server}:{imap_port}...")
    try:
        imap = imaplib.IMAP4_SSL(imap_server, imap_port)
        imap.login(username, password)
        
        # Select inbox and get email count
        imap.select('INBOX')
        status, messages = imap.search(None, 'ALL')
        email_count = len(messages[0].split()) if messages[0] else 0
        
        print(f"✅ IMAP connection successful!")
        print(f"📧 Found {email_count} emails in INBOX")
        
        # Test search for specific sender
        allowed_sender = 'dhananjayshahane24@gmail.com'
        status, messages = imap.search(None, f'FROM "{allowed_sender}"')
        filtered_count = len(messages[0].split()) if messages[0] else 0
        print(f"📬 Found {filtered_count} emails from {allowed_sender}")
        
        imap.close()
        imap.logout()
        
    except imaplib.IMAP4.error as e:
        print(f"❌ IMAP authentication failed: {str(e)}")
        print("💡 For Gmail, you may need to:")
        print("   1. Enable 2-factor authentication")
        print("   2. Generate an App Password")
        print("   3. Use the App Password instead of your regular password")
        return False
    except Exception as e:
        print(f"❌ IMAP connection failed: {str(e)}")
        return False
    
    # Test SMTP connection
    print(f"\n📤 Testing SMTP connection to {smtp_server}:{smtp_port}...")
    try:
        smtp = smtplib.SMTP(smtp_server, smtp_port)
        smtp.starttls()
        smtp.login(username, password)
        smtp.quit()
        print("✅ SMTP connection successful!")
        
    except Exception as e:
        print(f"❌ SMTP connection failed: {str(e)}")
        return False
    
    print("\n🎉 All email tests passed!")
    return True

if __name__ == "__main__":
    test_email_connection()
