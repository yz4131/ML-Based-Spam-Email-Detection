import boto3
import email
import json
import os
import io
import csv
from helper import one_hot_encode, vectorize_sequences
import numpy as np


# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    s3 = boto3.client('s3')

    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # retrive email
    response = s3.get_object(Bucket=bucket, Key=key)
    all_content = response['Body'].read().decode('utf-8')
    parsed = email.message_from_string(all_content)
    date = parsed['Date']
    subject = parsed['Subject']
    sender = parsed['From']
    sender = sender.split('<')[1][:-1]
    for part in parsed.walk():
        if part.get_content_type() == 'text/plain':
            content_temp = part.get_payload()
            break

    # create a sample
    length = len(content_temp)
    if length > 240:
        sample = content_temp[:240]
    else:
        sample = content_temp

    # remove \n
    content = ''
    for l in content_temp:
        if l != '\n' and l != '\r':
            content += l
    content = [content]

    # call sagemaker model endpoints
    # content = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"]
    vocabulary_length = 9013
    one_hot_content = one_hot_encode(content, vocabulary_length)
    encoded_content = vectorize_sequences(one_hot_content, vocabulary_length)

    payload = json.dumps(encoded_content.tolist())
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=payload)
    result = json.loads(response['Body'].read().decode("utf-8"))
    pred = int(result['predicted_label'][0][0])
    score = result['predicted_probability'][0][0]
    label = 'Spam' if pred == 1 else 'non-spam'

    #send reply
    reply = """<html>
               <head></head>
               <body>
                    <h1>Auto-Reply</h1>
                    <p>We received your email sent at "{}" with the subject "{}".
                       Here is a 240 character sample of the email body: </p>
                    <p>   {}  </p>
                    <p>
                       The email was categorized as "{}" with a "{}%" confidence.
                    </p>
               </body>
               </html>
            """.format(date,subject,sample,label,score*100)
    CHARSET = "UTF-8"
    SUBJECT = "Auto-Reply"
    ses = boto3.client('ses',region_name="us-east-1")
    resp = ses.send_email(
            Destination={
                'ToAddresses': [
                    sender,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': reply,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source='<no-reply@yutongc123.com>',
        )
    print(resp)
