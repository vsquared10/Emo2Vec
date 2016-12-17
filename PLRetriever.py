
# coding: utf-8

import requests

client_id = 'b8b2038593d44cf6b826d84a09238016'
client_key = '0d8de85a4bae4c519f809302af16f756'

def callAPI(mood):
    url= 'https://api.spotify.com/v1/search'
    params = {'q' : mood,
              'type' :'playlist',
              'limit' : '5'}
    headers={
        "Accept": "text/plain"
      }
    r = requests.get(url, params=params)
    data = r.json()
    return data
