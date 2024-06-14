import smtplib
import ssl

#setup port number and server name

smtp_port = 587 #standard secure SMTP port
smtp_server = "smtp.gmail.com" #Google SMTP server

email_sender = "hfjulien2001@graduate.utm.my"
email_receiver = "hfjulien2001@graduate.utm.my"

pswd = "npyfzrwzrvmsyzgp"
#cotents of message

message = "hi"

simple_email_context = ssl.create_default_context()

try:
    print("Connecting to server...")
    TIE_server = smtplib.SMTP(smtp_server, smtp_port)
    TIE_server.starttls(context=simple_email_context)
    TIE_server.login(email_sender, pswd)
    print("Connected to server!!")

    print()
    print(f"Sending email to {email_receiver}")
    TIE_server.sendmail(email_sender, email_receiver, message)
    print(f"Email successfully sent to {email_receiver}")

except Exception as e:
    print(e)

finally:
    TIE_server.quit()