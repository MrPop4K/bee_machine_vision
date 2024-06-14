import os
from email.message import EmailMessage
import ssl
import smtplib

sender_email = 'hfjulien2001@graduate.utm.my'
sender_password = 'npyf zrwz rvms yzgp'
receiver_email = 'renewsyndicate@gmail.com'

subject = 'Python test email'

body = """
test email kontol """

em = EmailMessage()
em['From'] = sender_email
em['To'] = receiver_email
em['Subject'] = subject
em.set_content(body)

context = ssl.create_default_context()

with smtplib.SMTP_SSL('smtp.gmail.com', 465 , context=context) as smtp:
    smtp.login(sender_email, sender_password)
    smtp.sendmail(sender_email, receiver_email, em.as_string())
