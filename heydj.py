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
from EmoClassifier import emo2vec, model
from PLRetriever import callAPI
from bot_responses import convos, reflections
import numpy as np
from nltk.chat import *
import os
import time
from slackclient import SlackClient

user_input = []
samples = []
cachedStopWords = stopwords.words("english")

def reflect(fragment):
    tokens = fragment.lower().split(' ')
    tokens = [*map(lambda x:reflections[x] if x in reflections else x, tokens)]
    return ' '.join(tokens)


def analyze(statement):
    """
    Match user's input to responses in psychobabble. Then reflect candidate response.
    If response is 'music4lyfe', then make a call to Spotify's API and return a playlist
    based on the OnevsRest(OVR) classifier's interpretation of user's input
    emotional content.
    """
    statement = statement.lower()
    for item in convos:
        match = re.search(item[0],statement)
        if match:
            response = np.random.choice(item[1])
            return response.replace('{0}',reflect(match.group(0)))
        elif match == "music4lyfe":
            samples = [word for word in user_input if word not in cachedStopWords]
            similar_word, similarity = model.most_similar(positive=samples, topn=1)[0]
            OVR_input = model[similar_word].reshape(1, -1)
            query = list(emotions.columns)[OVR.predict(OVR_input)]
            callAPI(query)
            response = "<" + data['playlists']['items'][0]['href'] + ">"
            return response

def conversation_start():
    print("Hello. How are you feeling today?")
    while True:
        statement = input("> ")
        print(analyze(statement))
        user_input.append(statement)
        user_input = " ".join(c.strip(string.punctuation) for c in user_input).lower().split(' ')

        # if statement == "MUSIC4LYFE":
        #     samples = [word for word in user_input if word not in cachedStopWords]
        #     similar_word, similarity = model.most_similar(positive=samples, topn=1)[0]
        #     OVR_input = model[similar_word].reshape(1, -1)
        #     query = emo2vec.label_name[OVR.predict(OVR_input)].values[0]
        #     callAPI(query)

        if statement == "quit":
            print(analyze(statement))
            break

# starterbot's ID as an environment variable
BOT_ID = os.environ.get("BOT_ID")

# constants
AT_BOT = "<@" + BOT_ID + ">"
EXAMPLE_COMMAND = "do"

# instantiate Slack & Twilio clients
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))

def handle_command(command, channel):
    """
        Receives commands directed at the bot and determines if they
        are valid commands. If so, then acts on the commands. If not,
        returns back what it needs for clarification.
    """

    response = analyze(command)
    slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, unfurl_links=True, as_user=True)

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
