import smtplib
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders
import os
import ssl

#setup port number and server name

smtp_port = 587 #standard secure SMTP port
smtp_server = "smtp.gmail.com" #Google SMTP server

email_sender = "hfjulien2001@graduate.utm.my"
email_list = ["hfjulien2001@graduate.utm.my", "hfjulien2001@graduate.utm.my"]

pswd = "npyfzrwzrvmsyzgp"

subject = "NEW EMAIL PHTHON"

def send_emails(email_list):

    for person in email_list:

        #Body of email
        body = """
        
        Hello my brudda
        wastup
        this is a scam
        
        """


        #make a MIME object to define parts of the email

        msg = MIMEMultipart()
        msg['From'] = email_sender
        msg['To'] = person
        msg['Subject'] = subject

        #Attach the body of the message
        msg.attach(MIMEText(body, 'plain'))

        #Define the file to attach
        filename = "E:\\video_for_yolo\\bee_01_counter_result.mp4"

        #Open the file in python as binary
        attachment = open(filename, 'rb') #r for read and b for binary

        #Encode as base 64
        attachment_package = MIMEBase('application', 'octet-stream')
        attachment_package.set_payload((attachment).read())
        encoders.encode_base64(attachment_package)
        attachment_package.add_header('Content-Disposition', "attachment; filename= " + filename)
        msg.attach(attachment_package)

        #Cast as string
        text = msg.as_string()

        #Connect with the server
        print("Connecting to server...")
        TIE_server = smtplib.SMTP(smtp_server, smtp_port)
        TIE_server.starttls()
        TIE_server.login(email_sender, pswd)
        print("Connected to server!!")
        print()

        #Send emails to "person" as list is iterated
        print(f"Sending email to {person}")
        TIE_server.sendmail(email_sender, person, text)
        print(f"Email successfully sent to {person}")
        print()

    TIE_server.quit()


send_emails(email_list)

# #cotents of message
#
# message = "hi"
#
# simple_email_context = ssl.create_default_context()
#
# try:
#     print("Connecting to server...")
#     TIE_server = smtplib.SMTP(smtp_server, smtp_port)
#     TIE_server.starttls(context=simple_email_context)
#     TIE_server.login(email_sender, pswd)
#     print("Connected to server!!")
#
#     print()
#     print(f"Sending email to {email_receiver}")
#     TIE_server.sendmail(email_sender, email_receiver, message)
#     print(f"Email successfully sent to {email_receiver}")
#
# except Exception as e:
#     print(e)
#
# finally:
#     TIE_server.quit()