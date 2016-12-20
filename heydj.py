#!/usr/bin/env python
#!/usr/bin/env python
""" A chatbot that listens to how peoples' days are going and
recommends a Spotify playlist based on it's interpretation of their mood.
"""

import re
import random
import string
import nltk
from nltk.corpus import stopwords
from EmoClassifier import emo2vec, model, emotions, emo_model
from PLRetriever import callAPI
from bot_responses import *
import numpy as np
from nltk.chat import *
import os
import time
from slackclient import SlackClient

samples = []
cachedStopWords = stopwords.words("english")

def reflect(fragment):
    tokens = fragment.lower().split(' ')
    tokens = [*map(lambda x:reflections[x] if x in reflections else x, tokens)]
    return ' '.join(tokens)

def playlist_recommender(list_of_strings):
    mood = " ".join("".join(char for char in sent if char not in string.punctuation) for sent in list_of_strings).lower().split(' ')
    prediction_words = [word for word in mood if word not in cachedStopWords]
    similar_word_vectors, degree_of_similarity = model.most_similar(positive=prediction_words, topn=1)[0]
    prediction = model[similar_word_vectors].reshape(1, -1)
    call = list(emotions.columns)[emo_model.predict(prediction)]
    data = callAPI(call)
    items = data['playlists']['items']
    playlist = random.choice(items)['external_urls']['spotify']
    return playlist

def analyze(statement):
    """
    Match user's input to responses in psychobabble. Then reflect candidate response.
    """
    statement = statement.lower()
    for item in convos:
        match = re.search(item[0],statement)
        if match:
            response = np.random.choice(item[1])
            return response.replace('{0}',reflect(match.group(0)))

# starterbot's ID as an environment variable
BOT_ID = os.environ.get("BOT_ID")

# constants
AT_BOT = "<@" + BOT_ID + ">"
REQUEST_COMMAND = "retrieval"

# instantiate Slack & Twilio clients
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))

def handle_command(command, channel):
    """
        Receives commands directed at the bot and determines if they
        are valid commands. If so, then acts on the commands. If not,
        returns back what it needs for clarification.
    """
    feelings_list = []
    feelings_list.append(command)
    
    response = "Hello Beautiful! How are you feeling today? Use the *" + REQUEST_COMMAND + \
               "* request to get my musical interpretation of your mood today."
    if command.startswith(REQUEST_COMMAND):
        slack_client.api_call("chat.postMessage", channel=channel,text="reading your mood...", as_user=True)
        response = "<" + playlist_recommender(feelings_list) + "|Click here!!!>"
        slack_client.api_call("chat.postMessage",
         channel=channel,
         text="...aaand here's your perfect playlist for this moment :)\n" + response + "\n SO FRESH! Listen and enjoy",
          unfurl_links=True, as_user=True)
    else:
        response3 = analyze(command)
        slack_client.api_call("chat.postMessage", channel=channel,text=response3, as_user=True)

def parse_slack_output(slack_rtm_output):
    """
        The Slack Real Time Messaging API is an events firehose.
        this parsing function returns None unless a message is
        directed at the Bot, based on its ID.
    """
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output and 'text' in output and AT_BOT in output['text']:
                # return text after the @ mention, whitespace removed
                return output['text'].split(AT_BOT)[1].strip().lower(), \
                       output['channel']

    return None, None


if __name__ == "__main__":
    READ_WEBSOCKET_DELAY = 1 # 1 second delay between reading from firehose
    if slack_client.rtm_connect():
        print("StarterBot connected and running!")
        while True:
            command, channel = parse_slack_output(slack_client.rtm_read())
            if command and channel:
                handle_command(command, channel)
            time.sleep(READ_WEBSOCKET_DELAY)
    else:
        print("Connection failed. Invalid Slack token or bot ID?")
