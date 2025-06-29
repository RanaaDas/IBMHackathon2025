import streamlit as st
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_rejection_email(recipient_email, subject, body):
    """Sends an email using SendGrid."""
    try:
        # Retrieve secrets from Streamlit's secrets.toml
        sendgrid_api_key = st.secrets["SENDGRID_API_KEY"]
        sender_email = st.secrets["SENDER_EMAIL"]

        # --- FIX: Convert newlines to <br/> for HTML formatting and remove <strong> ---
        # Also ensure leading/trailing whitespace is stripped for clean HTML
        html_body = body.replace("\n", "<br/>").strip()
        # The .replace("`","") is a small hack in case the LLM output includes backticks.
        html_body = html_body.replace("`","")
        # --- END FIX ---

        message = Mail(
            from_email=sender_email,
            to_emails=recipient_email,
            subject=subject,
            html_content=html_body # Use the correctly formatted HTML body
        )
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)

        if response.status_code == 202: # 202 means Accepted
            st.success(f"Email successfully sent to {recipient_email}!")
            return True
        else:
            st.error(f"Email sending failed (Status: {response.status_code}, Body: {response.body}). Please check SendGrid API key, sender email verification, and recipient address.")
            return False
    except KeyError as e:
        st.error(f"SendGrid configuration error: Missing secret '{e}'. Please check your .streamlit/secrets.toml file.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while sending email with SendGrid: {e}")
        return False