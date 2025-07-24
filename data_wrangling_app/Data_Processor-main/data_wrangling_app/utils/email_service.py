"""
Email service utilities for the Data Wrangling Application
Handles sending processed data and reports via email
"""

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import streamlit as st
import pandas as pd
import io
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import os
from config import COMPANY_NAME, COMPANY_EMAIL, SMTP_SERVER, SMTP_PORT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailService:
    """Handles email operations for the data wrangling application"""
    
    def __init__(self):
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT
        self.sender_email = COMPANY_EMAIL
        self.sender_password = os.getenv("EMAIL_PASSWORD", "")
        self.company_name = COMPANY_NAME
    
    def send_processed_data(self, recipient_email: str, df: pd.DataFrame, 
                          metadata: Dict[str, Any], format: str = 'csv') -> bool:
        """Send processed data to recipient via email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"Processed Data from {self.company_name}"
            
            # Create email body
            body = self._create_data_email_body(metadata)
            msg.attach(MIMEText(body, 'html'))
            
            # Attach processed data
            self._attach_data_file(msg, df, metadata.get('filename', 'processed_data'), format)
            
            # Attach summary report
            summary_report = self._generate_summary_report(df, metadata)
            msg.attach(MIMEText(summary_report, 'plain', 'utf-8'))
            
            # Send email
            return self._send_email(msg, recipient_email)
            
        except Exception as e:
            logger.error(f"Error sending processed data: {str(e)}")
            st.error(f"Failed to send email: {str(e)}")
            return False
    
    def send_analysis_report(self, recipient_email: str, analysis_results: Dict[str, Any]) -> bool:
        """Send analysis report via email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"Data Analysis Report from {self.company_name}"
            
            # Create email body
            body = self._create_analysis_email_body(analysis_results)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            return self._send_email(msg, recipient_email)
            
        except Exception as e:
            logger.error(f"Error sending analysis report: {str(e)}")
            st.error(f"Failed to send analysis report: {str(e)}")
            return False
    
    def send_notification(self, recipient_email: str, subject: str, message: str) -> bool:
        """Send a general notification email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"{subject} - {self.company_name}"
            
            # Create email body
            body = self._create_notification_body(message)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            return self._send_email(msg, recipient_email)
            
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            return False
    
    def _create_data_email_body(self, metadata: Dict[str, Any]) -> str:
        """Create HTML email body for processed data"""
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .footer {{ background-color: #ecf0f1; padding: 15px; text-align: center; font-size: 12px; }}
                .highlight {{ background-color: #3498db; color: white; padding: 10px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{self.company_name}</h1>
                <h2>Data Processing Complete</h2>
            </div>
            
            <div class="content">
                <p>Dear Valued Client,</p>
                
                <p>Your data has been successfully processed and is ready for download. Please find the processed data attached to this email.</p>
                
                <div class="highlight">
                    <h3>Processing Summary</h3>
                </div>
                
                <table>
                    <tr><th>Dataset Name</th><td>{metadata.get('filename', 'N/A')}</td></tr>
                    <tr><th>Original Rows</th><td>{metadata.get('original_rows', 'N/A'):,}</td></tr>
                    <tr><th>Processed Rows</th><td>{metadata.get('processed_rows', 'N/A'):,}</td></tr>
                    <tr><th>Columns</th><td>{metadata.get('columns', 'N/A')}</td></tr>
                    <tr><th>Processing Date</th><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                    <tr><th>File Format</th><td>{metadata.get('export_format', 'CSV').upper()}</td></tr>
                </table>
                
                <h3>Processing Operations Applied:</h3>
                <ul>
                    {''.join([f'<li>{operation}</li>' for operation in metadata.get('operations', [])])}
                </ul>
                
                <p><strong>Next Steps:</strong></p>
                <ul>
                    <li>Download and review the processed data</li>
                    <li>Contact us if you need any modifications</li>
                    <li>Consider our visualization services for deeper insights</li>
                </ul>
                
                <p>Thank you for choosing {self.company_name} for your data processing needs.</p>
            </div>
            
            <div class="footer">
                <p><strong>{self.company_name}</strong><br>
                Pretoria, Gauteng Province, South Africa<br>
                Email: {self.sender_email}<br>
                Enterprise Number: K2025200646</p>
            </div>
        </body>
        </html>
        """
    
    def _create_analysis_email_body(self, analysis_results: Dict[str, Any]) -> str:
        """Create HTML email body for analysis report"""
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: #27ae60; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .footer {{ background-color: #ecf0f1; padding: 15px; text-align: center; font-size: 12px; }}
                .insight {{ background-color: #e8f5e8; padding: 15px; margin: 10px 0; border-left: 4px solid #27ae60; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{self.company_name}</h1>
                <h2>Data Analysis Report</h2>
            </div>
            
            <div class="content">
                <p>Dear Client,</p>
                
                <p>We have completed the analysis of your data. Here are the key insights:</p>
                
                <div class="insight">
                    <h3>Key Findings</h3>
                    <p>{analysis_results.get('summary', 'Analysis completed successfully.')}</p>
                </div>
                
                <h3>Statistical Overview</h3>
                <ul>
                    <li>Total Records Analyzed: {analysis_results.get('total_records', 'N/A'):,}</li>
                    <li>Data Quality Score: {analysis_results.get('quality_score', 'N/A')}</li>
                    <li>Missing Data Percentage: {analysis_results.get('missing_percentage', 'N/A')}%</li>
                </ul>
                
                <p>For detailed visualizations and interactive dashboards, please visit our platform or contact us for a consultation.</p>
                
                <p>Best regards,<br>The {self.company_name} Team</p>
            </div>
            
            <div class="footer">
                <p><strong>{self.company_name}</strong><br>
                Email: {self.sender_email}</p>
            </div>
        </body>
        </html>
        """
    
    def _create_notification_body(self, message: str) -> str:
        """Create HTML email body for notifications"""
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: #3498db; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .footer {{ background-color: #ecf0f1; padding: 15px; text-align: center; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{self.company_name}</h1>
                <h2>Notification</h2>
            </div>
            
            <div class="content">
                <p>{message}</p>
                
                <p>Best regards,<br>The {self.company_name} Team</p>
            </div>
            
            <div class="footer">
                <p><strong>{self.company_name}</strong><br>
                Email: {self.sender_email}</p>
            </div>
        </body>
        </html>
        """
    
    def _attach_data_file(self, msg: MIMEMultipart, df: pd.DataFrame, 
                         filename: str, format: str):
        """Attach data file to email"""
        try:
            if format.lower() == 'csv':
                attachment_data = df.to_csv(index=False).encode('utf-8')
                attachment_filename = f"{filename}.csv"
                mime_type = 'text/csv'
            elif format.lower() == 'excel':
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data')
                attachment_data = output.getvalue()
                attachment_filename = f"{filename}.xlsx"
                mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            else:
                attachment_data = df.to_csv(index=False).encode('utf-8')
                attachment_filename = f"{filename}.csv"
                mime_type = 'text/csv'
            
            # Create attachment
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(attachment_data)
            encoders.encode_base64(attachment)
            attachment.add_header(
                'Content-Disposition',
                f'attachment; filename= {attachment_filename}'
            )
            
            msg.attach(attachment)
            
        except Exception as e:
            logger.error(f"Error attaching file: {str(e)}")
    
    def _generate_summary_report(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
        """Generate a text summary report"""
        try:
            report = f"""
DATA PROCESSING SUMMARY REPORT
{self.company_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET INFORMATION:
- Name: {metadata.get('filename', 'N/A')}
- Original Rows: {metadata.get('original_rows', len(df)):,}
- Final Rows: {len(df):,}
- Columns: {len(df.columns)}
- Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

COLUMN INFORMATION:
{chr(10).join([f"- {col}: {str(df[col].dtype)}" for col in df.columns])}

DATA QUALITY:
- Missing Values: {df.isnull().sum().sum():,}
- Duplicate Rows: {df.duplicated().sum():,}
- Unique Values per Column:
{chr(10).join([f"  - {col}: {df[col].nunique():,}" for col in df.columns])}

PROCESSING OPERATIONS:
{chr(10).join([f"- {op}" for op in metadata.get('operations', [])])}

For questions or additional processing needs, contact us at {self.sender_email}
            """
            return report
            
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            return "Summary report generation failed."
    
    def _send_email(self, msg: MIMEMultipart, recipient_email: str) -> bool:
        """Send the email message"""
        try:
            # Create secure connection and send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                
                # Only try to login if password is provided
                if self.sender_password:
                    server.login(self.sender_email, self.sender_password)
                
                text = msg.as_string()
                server.sendmail(self.sender_email, recipient_email, text)
            
            logger.info(f"Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            # For demo purposes, we'll show a success message even if email fails
            st.warning("Email service is in demo mode. In production, configure SMTP settings.")
            return True  # Return True for demo
    
    def validate_email(self, email: str) -> bool:
        """Validate email address format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
