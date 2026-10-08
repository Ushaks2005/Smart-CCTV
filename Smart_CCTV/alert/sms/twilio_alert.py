from twilio.rest import Client
import logging

# 🧱 Twilio Configuration (Replace with your credentials)
ACCOUNT_SID = "your_account_sid_here"
AUTH_TOKEN = "your_auth_token_here"
FROM_PHONE = "+1xxxxxxxxxx"   # Twilio verified phone number
TO_PHONE = "+91xxxxxxxxxx"    # Your mobile number

def send_twilio_alert(message):
    """
    Sends an SMS alert using Twilio API.
    """
    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=FROM_PHONE,
            to=TO_PHONE
        )
        logging.info(f"📱 Twilio alert sent: {message}")
    except Exception as e:
        logging.error(f"❌ Failed to send Twilio alert: {e}")
