import os
import re
from typing import List

import numpy as np
import stanza
import re
import pandas as pd
import snscrape.modules.twitter as sntwitter
from datasets import load_dataset
from wordfreq import tokenize, word_frequency
from xml.dom.minidom import parse
import xml.dom.minidom
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from tqdm import tqdm

# from emoji import is_emoji

# from transformers import pipeline

# ner = pipeline('ner', model='clarin-pl/FastPDN', aggregation_strategy='simple')

stanza.download('pl')

nlp = stanza.Pipeline(lang='pl', processors='tokenize,mwt,pos,lemma', use_gpu=True)

PATH_TO_RAW_DATA = "../data/full_corpus/"


def prepare_oscar():
    """Shard and serialize shuffled part of web dump in Polish language;
    Contains 500 thousand samples.
    :return: None
    """
    oscar_ds = load_dataset("oscar", "unshuffled_deduplicated_pl")
    oscar_ds = oscar_ds.shuffle(seed=42)
    oscar_shard = oscar_ds['train'].shard(num_shards=160, index=0)
    oscar_shard.to_csv("../data/full_corpus/oscar.csv")


def prepare_twitter():
    query = 'lang:pl'
    limit = 1000000
    result = []
    tweets = sntwitter.TwitterSearchScraper(query).get_items()
    for i, tweet in enumerate(tweets):
        if i > limit:
            break
        result.append([tweet.content, tweet.username])
    tweets_df = pd.DataFrame(result, columns=['text', 'username'])
    # tweets_df['text'] = tweets_df['text'].apply(lambda x: clean_tweet(x))
    tweets_df.to_csv("/home/ndazhunts/CLARIN/stylometry/stylometry/data/full_corpus/tweets.csv")


def clean_tweet(tweet: str):
    # remove hyperlinks with http(s)
    tweet = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)",
                   '', tweet, flags=re.MULTILINE)
    # remove other URLs
    tweet = re.sub(r'[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)',
                   '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r"http\S+", "", tweet)
    # remove mentions, hashtags and emojis
    tweet_split = tweet.split(' ')
    tweet_split = [word for word in tweet_split if not is_emoji(word)
                   and not word.startswith('@')
                   and not word.startswith('#')
                   and word.isalnum()]
    tweet = ' '.join(tweet_split)
    return tweet


def prepare_tweets():
    path = "../data/full_corpus/tweets.csv"
    tweets_df = pd.read_csv(path)
    tweets_df = tweets_df.drop('Unnamed: 0', axis=1).fillna('')
    tweets_df["text"] = tweets_df["text"].apply(lambda x: clean_tweet(x))
    tweets_df.to_csv("../data/full_corpus/tweets.csv")


def prepare_wiki():
    path = "../data/full_corpus/plwiki3"
    texts = []
    for letter in os.listdir(path):
        for number in os.listdir(os.path.join(path, letter)):
            for article in os.listdir(os.path.join(path, letter, number)):
                path_to_text = os.path.join(path, letter, number, article)
                if os.path.isfile(path_to_text):
                    with open(path_to_text, 'r') as text:
                        lines = text.readlines()
                        lines = [line.strip() for line in lines if "<" not in line]
                        lines = [line for line in lines if len(line) > 0]
                        result = ' '.join(lines)
                        texts.append(result)
    result = pd.DataFrame(texts)
    result.columns = ['text']
    result.to_csv("../data/full_corpus/wiki.csv")


def prepare_nkjp():
    authors = prepare_nkjp_authors()
    texts = prepare_nkjp_sentences()
    merged = [dict(sent, **author) for sent, author in zip(texts, authors)]
    df = pd.DataFrame(merged)
    df.to_csv("../data/full_corpus/nkjp.csv")
    # # result.columns = ['text']


def prepare_nkjp_sentences():
    sentences = []
    data_path = "../data/raw/nkjp/NKJP-PodkorpusMilionowy-1.2"
    for filedir in os.listdir(data_path):
        text_dict = {"file_id": filedir}
        file = os.path.join(data_path, filedir, "text.xml")
        DOMTree = xml.dom.minidom.parse(file)
        chunk_list = DOMTree.documentElement
        divs = chunk_list.getElementsByTagName("div")
        for div in divs:
            sents = div.getElementsByTagName("ab")
            sents = [sent.childNodes[0].data for sent in sents]
            text_dict["texts"] = sents
        sentences.append(text_dict)
    return sentences


def prepare_nkjp_authors():
    data_path = "../data/raw/nkjp/NKJP-PodkorpusMilionowy-1.2"
    authors = []
    for filedir in os.listdir(data_path):
        author_dict = {'file_id': filedir}
        header = os.path.join(data_path, filedir, "header.xml")
        DOMTree_auth = xml.dom.minidom.parse(header)
        chunk_list_auth = DOMTree_auth.documentElement
        author_node = chunk_list_auth.getElementsByTagName("author")
        if author_node and author_node[0].firstChild:
            author = author_node[0].firstChild.nodeValue
        else:
            author = "Nie znany"
        author_dict["author"] = author
        authors.append(author_dict)
    return authors


# def extract_ner(author: str) -> List[str]:
#     ner_res = ner(author)
#     return [output['entity_group'] for output in ner_res]


# def prepare_nkjp_ner(df: pd.DataFrame):
#     path_nkjp = "../data/full_corpus/nkjp.csv"
#     df = pd.read_csv(path_nkjp)
#     df_authors = df[df.author != 'Nie znany']
#     df_authors["ner"] = df_authors["author"].apply(lambda x: extract_ner(x))
#     path_out = "../data/full_corpus/nkjp_authors.csv"
#     df_authors.to_csv(path_out)


def combine_corpus():
    pass


def serialize_corpus():
    pass


def serialize_corpus_tokens(dataset_df: pd.DataFrame, out_name) -> None:
    texts = dataset_df['text'].to_numpy()
    tokens = []
    for text in texts:
        tokenized = tokenize(text=text, lang='pl')
        tokens.extend(tokenized)
    result = pd.DataFrame(tokens)
    result.columns = ['token']

    def has_numbers(inputString):
        return any(char.isdigit() for char in inputString)

    mask = (result['token'].apply(lambda x: not has_numbers(x)))
    result = result.loc[mask]
    result.to_csv(out_name)


def clean_frequencies(tokens_df):
    tokens_df = tokens_df.loc[tokens_df.token.str.contains('^[a-zA-Z][a-zA-Z, ]*$')]
    tokens_df.to_csv("../data/full_corpus/oscar_tokens_cleaned.csv")


def lemmatize_token(token):
    doc = nlp(token)
    lemmas = []
    for sent in doc.sentences:
        for word in sent.words:
            lemmas.append(word.lemma)
    return lemmas


def lemmatize_text(dataset_df, path_out):
    tokens = list(set(dataset_df['text'].tolist()))
    lemmas = [lemmatize_token(token=word) for word in tokens]
    assert len(tokens) == len(lemmas)
    dataset_df['lemma'] = lemmas
    dataset_df.to_csv(path_out)



def remove_short_tweets():
    path = "../data/full_corpus/tweets.csv"
    df = pd.read_csv(path)
    mask = (df['text'].apply(lambda x: len(str(x).split(' ')) > 2))
    df = df.loc[mask]
    df.to_csv(path)

def remove_unnecessary_authors():
    path = "../data/full_corpus/tweets_pl.csv"
    df = pd.read_csv(path)
    authors_count = df[df.groupby('username').username.transform('count') >= 30].copy()
    for col in authors_count.columns:
        if 'Unnamed' in col:
            authors_count = authors_count.drop(col, axis=1)
    authors_count.to_csv(path)

def prepare_tokens(path_to_raw: str) -> None:
    full_path = os.path.join(PATH_TO_RAW_DATA, path_to_raw)
    texts_df = pd.read_csv(full_path)


def get_global_frequencies(tokens_file: str) -> pd.DataFrame:
    path_to_ds = os.path.join(PATH_TO_RAW_DATA, tokens_file)
    tokens_df = pd.read_csv(path_to_ds)
    tokens_df["global_freq"] = tokens_df["tokens"].apply(lambda x: word_frequency(word=x, lang="pl"))
    return tokens_df


def get_local_frequencies(tokens_file: str, path_out: str) -> None:
    path_to_ds = os.path.join(PATH_TO_RAW_DATA, tokens_file)
    tokens_df = pd.read_csv(path_to_ds)
    print(tokens_df.columns)
    print(tokens_df.head())
    frequencies = tokens_df.groupby(['token'])['token'].count()
    print(frequencies.head())
    frequencies.to_csv(os.path.join(PATH_TO_RAW_DATA, path_out))

    #
    # for word in tokens:
    #     occurrence = tokens.count(word)
    #     frequencies[word] = occurrence / corpus_size
    # result = pd.DataFrame.from_dict(data=frequencies, orient='index').reset_index()
    # result.columns = ["token", "local_frequency"]
    # result["general_frequency"] = result["token"].apply(lambda x: word_frequency(word=x, lang='pl'))
    # result.to_csv(os.path.join(PATH_TO_RAW_DATA, "oscar_stats.csv"))
    # # print(result.shape)
    # # print(result.head())


if __name__ == '__main__':
    # # prepare_oscar()
    # path_to_oscar = os.path.join(PATH_TO_RAW_DATA, "oscar_tokens_cleaned.csv")
    # oscar_df = pd.read_csv(path_to_oscar)
    # print(oscar_df.shape)
    # prepare_nkjp()
    # oscar_df = oscar_df.fillna("100")
    # print(oscar_df['token'].isnull().values.any())
    # # serialize_corpus_tokens(oscar_df, "oscar_tokens.csv")
    # # get_word_frequencies('oscar_tokens.csv')
    # clean_frequencies(oscar_df)
    # lemmatize_text(oscar_df)
    # nkjp = pd.read_csv("/home/ndazhunts/CLARIN/stylometry/stylometry/data/full_corpus/nkjp_authors.csv")
    # print(nkjp['ner'].unique())
    # prepare_twitter()
    # prepare_tweets()
    # remove_short_tweets()
    # tweets_df = pd.read_csv(r"C:\SI22_2\NLP\dataset\stylometry\data\full_corpus\tweets.csv")
    #
    # cols = [col for col in tweets_df.columns if "Unnamed" in col]
    # for col in cols:
    #     tweets_df = tweets_df.drop(col, axis=1)
    # in_path = "../data/full_corpus/tweets_pl.csv"
    # df = pd.read_csv(in_path)
    prepare_wiki()
    # print(df.columns)
    # out_path_tokens = "../data/full_corpus/twitter_tokens.csv"
    # serialize_corpus_tokens(dataset_df=df, out_name=out_path_tokens)
    out_path_lemmas = "../data/full_corpus/tweets_pl_lemmas.csv"
    # tokens_df = pd.read_csv(out_path_tokens)
    # lemmatize_text(dataset_df=df, path_out=out_path_lemmas)
    # remove_unnecessary_authors()
    # n = 500000
    # dfs = []
    # for g, df_chunk in df.groupby(np.arange(len(df)) // n):
    #     dfs.append(df_chunk)
    # print(len(dfs))
    # out_path = "../data/full_corpus/lemmas_twitter/"
    # for i, chunk in enumerate(tqdm(dfs)):
    #     filename = ''.join(["twitter_lemma", str(i), ".csv"])
    #     lemmatize_text(chunk, os.path.join(out_path, filename))
    #     print(f"{filename} serialized!")

    # lemmatize_text(tokens_df=df, path_out=out_path)
    # print(len(tweets_df['username'].unique()))
    # print(len(tweets_df['username']))
    # stats = tweets_df.groupby(['username'])['username'].count()
    # print(stats)
