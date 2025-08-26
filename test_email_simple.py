
#!/usr/bin/env python3
"""
Simple email test script to debug email issues
"""

import os
import imaplib
import smtplib
import email
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email import utils as email_utils

def test_email_simple():
    """Simple email test"""
    load_dotenv()
    
    username = os.getenv('EMAIL_USER')
    password = os.getenv('EMAIL_PASSWORD')
    
    print(f"🔧 TESTING EMAIL FUNCTIONALITY")
    print(f"📧 Username: {username}")
    print(f"🔑 Password: {'*' * len(password) if password else 'NOT SET'}")
    
    if not username or not password:
        print("❌ Credentials not found in .env file")
        return
    
    # Test IMAP - receiving emails
    print(f"\n📨 TESTING IMAP (RECEIVING)...")
    try:
        imap = imaplib.IMAP4_SSL('imap.gmail.com', 993)
        imap.login(username, password)
        print("✅ IMAP connection successful")
        
        imap.select('INBOX')
        status, messages = imap.search(None, 'ALL')
        if status == 'OK' and messages[0]:
            total_emails = len(messages[0].split())
            print(f"📬 Total emails in inbox: {total_emails}")
            
            # Get last 3 emails to see who they're from
            email_ids = messages[0].split()[-3:]
            print(f"\n📋 RECENT EMAILS:")
            for i, email_id in enumerate(reversed(email_ids), 1):
                try:
                    status, msg_data = imap.fetch(email_id, '(RFC822)')
                    if status == 'OK':
                        email_message = email.message_from_bytes(msg_data[0][1])
                        sender = email_message.get('From', 'Unknown')
                        subject = email_message.get('Subject', 'No Subject')
                        print(f"   {i}. From: {sender}")
                        print(f"      Subject: {subject}")
                except Exception as e:
                    print(f"   Error parsing email {i}: {e}")
        
        imap.close()
        imap.logout()
        
    except Exception as e:
        print(f"❌ IMAP test failed: {e}")
    
    # Test SMTP - sending emails
    print(f"\n📤 TESTING SMTP (SENDING)...")
    try:
        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = username  # Send to self for testing
        msg['Subject'] = "MCP Email Test - " + email_utils.formatdate(localtime=True)
        msg['Date'] = email_utils.formatdate(localtime=True)
        
        body = f"""This is a test email from the MCP Email Enhanced Server.
        
        Sent at: {email_utils.formatdate(localtime=True)}
        From server: MCP Enhanced Email Server
        
        If you receive this, email sending is working correctly!"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(username, password)
        server.send_message(msg)
        server.quit()
        
        print("✅ Test email sent successfully to yourself!")
        print(f"📧 Check your inbox for: {msg['Subject']}")
        
    except Exception as e:
        print(f"❌ SMTP test failed: {e}")
    
    print(f"\n✨ Email test completed!")

if __name__ == "__main__":
    test_email_simple()
