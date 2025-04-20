import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(to_email, subject, message):
    from_email = "shresthvatsal@gmail.com"
    to_email = "vatsalshresth@gmail.com"
    password = "aqzu hsox hdvw ipwg"  # Use App Password if you're using Gmail

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(message, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, password)
            server.send_message(msg)
            print("✅ Alert sent successfully!")
    except Exception as e:
        print("❌ Failed to send alert:", e)
