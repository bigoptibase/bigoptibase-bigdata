#! /usr/bin/env python3
# _*_ coding: utf8 _*_
#
# KafkaProducer
# from: python-kafka-doc
# from: file:///usr/share/doc/python-kafka-doc/html/usage.html
# from https://stackoverflow.com/questions/64857884/converting-from-python-kafka-to-confluent-kafka-how-to-create-parity-with-sasl

import sys, requests, json, logging, time
from kafka import KafkaProducer
from kafka.errors import KafkaError
from sys import exit

class TokenProvider(object):
   def __init__(self, client_id, client_secret, grant_type):
       self.client_id = client_id
       self.client_secret = client_secret
       self.token_req_payload = {'grant_type': grant_type}
   def token(self):
       token_response = requests.post(token_url,
                                      data=self.token_req_payload, verify=False, allow_redirects=False,
                                      auth=(client_id, client_secret))
       if token_response.status_code != 200:
           print("Failed to obtain token from the OAuth 2.0 server", file=sys.stderr)
           sys.exit(1)
       print("TokenProvider: Successfully obtained a new token!")
       tokens = json.loads(token_response.text)
       token = tokens['access_token']
       # print(token)
       return token

def on_send_success(record_metadata):
    print ('record_metadata.topic: ', record_metadata.topic)
    print ('record_metadata.partition', record_metadata.partition)
    print ('record_metadata.offset', record_metadata.offset)

def on_send_error(excp):
    log.error('I am an errback', exc_info=excp)
    # handle exception

print('producer-kafka: Starting...')

# Change this configuration
topic        = ''
payload      = ''
client_id    = ''
client_secret= ''
token_url    = 'https://<keycloak_ip>:8443/auth/realms/test_realm/protocol/openid-connect/token'
kafka_brokers= '<broker_ip>:9093'
ssl_cafile   = 'CARoot.pem'
ssl_certfile = 'certificate.pem'
ssl_keyfile  = 'key.pem'

print('producer-kafka: Establishing connection...')

producer = KafkaProducer(
    bootstrap_servers=[kafka_brokers],
    security_protocol='SASL_SSL',
    sasl_mechanism='OAUTHBEARER',
    sasl_oauth_token_provider=TokenProvider(client_id, client_secret, 'client_credentials'),
    ssl_check_hostname=False,
    ssl_cafile=ssl_cafile,
    ssl_certfile=ssl_certfile,
    ssl_keyfile=ssl_keyfile,
    value_serializer=lambda m: json.dumps(m).encode('ascii')
)

print('producer-kafka: Sending message...')
future = producer.send(topic, {'message_type':topic, 'slug':'bigoptibase', 'payload':payload})

# Block for 'synchronous' sends
try:
    record_metadata = future.get(timeout=10)
except KafkaError:
    # Decide what to do if produce request failed...
    log.exception()
    pass

# Successful result returns assigned partition and offset
on_send_success(record_metadata)

exit(10)

