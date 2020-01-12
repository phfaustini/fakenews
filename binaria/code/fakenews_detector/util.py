import os
import json
from glob import glob
from fileinput import input
from statistics import mean, stdev

import pandas as pd
from TwitterAPI import TwitterAPI


if os.path.isfile('credentials.txt'):
    f = input('credentials.txt')
    consumer_key = f.readline().rstrip()
    consumer_secret = f.readline().rstrip()
    access_token_key = f.readline().rstrip()
    access_token_secret = f.readline().rstrip()
    f.close()
    api = TwitterAPI(consumer_key, consumer_secret, auth_type='oAuth2')
else:
    pass
    #print("_get_credentials - There is no file")


def all_txt_into_queryfolders(platform: str, dataset: str, class_label: str):
    """Every .txt must be inside a query folder. Some datasets just provide
    a bunch of .txt mixed all together inside the same folder.
    In these cases, a folder is created for each .txt.

    :param platform: either 'Twitter', 'Website' or 'WhatsApp'
    :dataset: 'tweets_br', 'br'
    :param class_label: either 'True' of 'False'
    """
    for filepath in glob('datasets/{0}/{1}/Raw/{2}/*'.format(platform, dataset, class_label)):
        filename = filepath.split("/")[-1]
        queryfoldername = filename.split(".")[0]
        queryfolder = 'datasets/{0}/{1}/Raw/{2}/{3}/'.format(platform, dataset, class_label, queryfoldername)
        if not os.path.isdir(queryfolder):
            os.mkdir(queryfolder)
            os.rename(filepath, "{0}{1}".format(queryfolder, filename))


def whole_csv_to_txts(platform='Websites', dataset='Bhattacharjee', filename='fake_or_real_news_NONEWLINES.csv'):
    """
    Converts each row in fake_or_real_news.csv
    to a single .txt file. Files are saved into
    False or True folders, according to their labels.
    """
    filepath = "datasets/{0}/{1}/Raw/{2}".format(platform, dataset, filename)
    data = pd.read_csv(filepath)
    c_f = 0
    c_t = 0
    for index, row in data.iterrows():
        csvid = row['id']
        title = row['title']
        text = row['text']
        label = row['label']

        if len(text) < 280:
            continue

        if label == 'FAKE':
            c_f += 1
            target = "datasets/{0}/{1}/Raw/False/{2}.txt".format(platform, dataset, c_f)
        elif label == 'REAL':
            c_t += 1
            target = "datasets/{0}/{1}/Raw/True/{2}.txt".format(platform, dataset, c_t)
        else:
            print(row)
        with open(target, 'w') as f:
                f.write(text)


def tweetid_to_txt_rumdect(platform='Twitter', dataset='rumdect', filename='Twitter.txt'):
    """
    rumdect: tweets are in a giant .txt file in the format
    eventID label:0 id1 id2 id3 ...

    It creates, for each event, a folder under the correct
    label (True or False). Inside each folder event, there is
    a .txt named after its id.
    """
    filepath = "datasets/{0}/{1}/Raw/{2}".format(platform, dataset, filename)
    for line in input(filepath):
        line = line.split('\t')
        event = line[0].split(':')[-1]
        label = "False" if line[1].split(':')[-1] == '0' else "True"
        tweets = line[2].split(' ')[0:-1]
        for tweet in tweets:
            if not os.path.isdir("datasets/{0}/{1}/Raw/{2}/{3}/".format(platform, dataset, label, event)):
                os.mkdir("datasets/{0}/{1}/Raw/{2}/{3}/".format(platform, dataset, label, event))
            target = "datasets/{0}/{1}/Raw/{2}/{3}/{4}.txt".format(platform, dataset, label, event, tweet)
            with open(target, 'w') as f:
                    f.write(tweet)


def parse_acl(platform='Twitter', dataset='twitter15', filename='label.txt'):
    filepath = "datasets/{0}/{1}/{2}".format(platform, dataset, filename)
    for line in input(filepath):
        line = line.rstrip().split(":")
        id = line[1]
        if line[0].rstrip() == "false":
            label = "False"
        elif line[0].rstrip() == "true":
            label = "True"
        else:
            continue
        target = "datasets/{0}/{1}/Raw/{2}/{3}.txt".format(platform, dataset, label, id)
        if not os.path.exists(target):
            tweet = get_tweet_byid(id)
            if tweet != {}:
                with open(target, 'w') as f:
                    f.write(json.dumps(tweet))


def get_tweet_byid(tweetid: str) -> dict:
    """Retrieve a single tweet data via its id

    :param userid: a string with the id of a tweet.
    :return: json (dict) representation of a tweet.
    """
    item = {}
    if api is not None:
        try:
            api_response = api.request('statuses/show/:{0}'.format(tweetid), {'tweet_mode': 'extended'})
            if api_response.status_code == 200:
                for item in api_response.get_iterator():
                    print("Got {0}".format(tweetid))
            else:
                print("get_tweet_byid - {0}".format(api_response.text))
                item = {}
        except:
            print("get_tweet_byid - exception in api_response!")
            item = {}
    else:
        print("get_tweet_byid - self.api is None!")
        item = {}
    return item


def get_tweets_rumdect(platform='Twitter', dataset='rumdect', filename='Twitter.txt'):
    """Get the list of tweet_ids and downloads the tweets.
    Then, it renames the .txt for .xtx. The tweets in json
    format are saved with .tweet extension
    """
    while True:
        tweets_ids = "datasets/{0}/{1}/Raw/*/*/*.txt".format(platform, dataset)
        tweets_ids_list = glob(tweets_ids)
        for tweet in tweets_ids_list:
            tweet_id = tweet.split("/")[-1].split(".")[0]
            tweet_json = get_tweet_byid(tweet_id)
            if len(tweet_json) == 0:  # Error downloading it
                print("Empty json")
            else:
                label = tweet.split("/")[4]
                event = tweet.split("/")[5]
                filename = tweet.split("/")[6].split('.')[0]
                target = "datasets/{0}/{1}/Raw/{2}/{3}/{4}.tweet".format(platform, dataset, label, event, filename)
                old_name = tweet.replace("txt", "xtx")
                with open(target, 'w') as f:
                    f.write(json.dumps(tweet_json))
                print("Renaming to {0} after getting {1}".format(old_name, target))
                os.rename(tweet, old_name)


def statistics_dataset(dataset: str):
    folders_false = glob('{0}/Structured/False/*'.format(dataset))
    folders_true = glob('{0}/Structured/True/*'.format(dataset))
    folders = folders_false + folders_true

    elements = []

    for folder in folders:
        elements.append(len(glob(folder+"/*")))

    number_folders = len(folders)
    print("Number of topics = {0}".format(number_folders))
    print("Mean elements per folder = {0}".format(mean(elements)))
    print("Std elements per folder = {0}".format(stdev(elements)))
    print()


#statistics_dataset(dataset='datasets/Twitter/tweets_br')
# whole_csv_to_txts(platform='Websites', dataset='Bhattacharjee', filename='fake_or_real_news.csv')
# parse_acl()


#whole_csv_to_txts()
#all_txt_into_queryfolders(platform='Websites', dataset='Bhattacharjee', class_label='False')
#all_txt_into_queryfolders(platform='Websites', dataset='Bhattacharjee', class_label='True')

# tweetid_to_txt_rumdect()
#get_tweets_rumdect()
# get_tweet_byid("676016912044789760")
