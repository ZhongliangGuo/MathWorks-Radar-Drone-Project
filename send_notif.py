import http.client
import urllib

import torch
from argparse import ArgumentParser

class Client:
    def __init__(self, user_key='ukx16k98erkfedkw2p1avx1fq3yvhg', api_token='acwwzstipcwn9o3cxgosbf7iipiefn'):
        self.user_key = user_key
        self.api_token = api_token
        self.conn = http.client.HTTPSConnection("api.pushover.net:443")

    def send_msg(self, text):
        self.conn.request("POST", "/1/messages.json",
                          urllib.parse.urlencode({
                              "token": self.api_token,
                              "user": self.user_key,
                              "message": text,
                          }), {"Content-type": "application/x-www-form-urlencoded"})
        self.conn.getresponse()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--msg', type=str, default='')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    client = Client()
    client.send_msg(f'{torch.cuda.get_device_name(device)} finished workflow: {args.msg}.')
