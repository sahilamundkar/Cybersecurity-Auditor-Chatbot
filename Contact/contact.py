import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Function to send email
def send_email(to_email, subject, message, from_email, password):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    try:
        # Setup the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

# Streamlit form
st.title("Contact Us")

with st.form(key='contact_form'):
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")
    submit_button = st.form_submit_button(label='Send')

if submit_button:
    if name and email and message:
        from_email = "sherwin@intersources.inc"  # Replace with your email address
        password = "sherwin@2000"           # Replace with your email account password
        to_email = "techballoon9@gmail.com"  # Replace with the recipient's email address
        subject = f"Contact Us Form Submission from {name}"
        body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"

        if send_email(to_email, subject, body, from_email, password):
            st.success("Your message has been sent successfully!")
        else:
            st.error("There was an error sending your message. Please try again later.")
    else:
        st.error("Please fill out all fields.")
