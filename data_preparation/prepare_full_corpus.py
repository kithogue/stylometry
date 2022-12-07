import os
import re
from typing import List
import stanza
import re
import pandas as pd
import snscrape.modules.twitter as sntwitter
from datasets import load_dataset
from wordfreq import tokenize, word_frequency
from xml.dom.minidom import parse
import xml.dom.minidom
from sklearn.model_selection import train_test_split
# from transformers import pipeline

# ner = pipeline('ner', model='clarin-pl/FastPDN', aggregation_strategy='simple')

# stanza.download('pl')

# nlp = stanza.Pipeline(lang='pl', processors='tokenize,mwt,pos,lemma')

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
    # remove mentions, hashtags and emojis
    tweet = ''.join(re.sub(r"(@[A-Za-z0–9]+)|([0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    return tweet


def prepare_wiki():
    pass


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
    result.to_csv(os.path.join(PATH_TO_RAW_DATA, out_name))


def clean_frequencies(tokens_df):
    tokens_df = tokens_df.loc[tokens_df.token.str.contains('^[a-zA-Z][a-zA-Z, ]*$')]
    tokens_df.to_csv("../data/full_corpus/oscar_tokens_cleaned.csv")


# def lemmatize_token(token):
#     doc = nlp(token)
#     lemmas = []
#     for sent in doc.sentences:
#         for word in sent.words:
#             lemmas.append(word.lemma)
#     return lemmas[0]

#
# def lemmatize_text(tokens_df):
#     tokens_df['token'] = tokens_df['token'].apply(lambda x: lemmatize_token(x))
#     tokens_df.to_csv("../data/full_corpus/tokens_lemmatized.csv")


def get_word_frequencies(tokens_file: str) -> None:
    path_to_ds = os.path.join(PATH_TO_RAW_DATA, tokens_file)
    tokens_df = pd.read_csv(path_to_ds)
    print(tokens_df.columns)
    print(tokens_df.head())
    frequencies = tokens_df.groupby(['token'])['token'].count()
    print(frequencies.head())
    frequencies.to_csv("../data/test.csv")

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
    prepare_twitter()
