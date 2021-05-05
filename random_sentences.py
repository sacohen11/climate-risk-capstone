#################################################################
# Title: Random Sentences
# Author: Sam Cohen

# Notes:
# This function creates a dataset of random sentences to use to
# fine-tune the model.
##################################################################

# Packages
import numpy as np
import pandas as pd
import newspaper
from newspaper import Article
import nltk
nltk.download('punkt')

def random_sentences():
    """
    This function uses the Newspaper3k package (https://newspaper.readthedocs.io/en/latest/) to download text from CNN articles.
    The first, second, and last sentence from each article are taken and stored.
    In addition, 100 other random sentences hand-picked by me from random websites are added.
    This new dataset of random sentences is saved to be used later.
    :return: nothing, but a dataset of random sentences is outputted
    """
    # First bit, extract the random sentences from the CNN articls

    urls = pd.read_excel("training_sentences.xlsx", sheet_name="Sheet2", index_col=0)
    sentences = []

    for url in urls["url"]:
        #url = "https://www.cnn.com/2021/02/12/tech/facebook-myanmar-military-intl-hnk/index.html"
        print(url)
        article = Article(url)
        article.download()
        try:
            article.parse()
            last_index = len(nltk.sent_tokenize(article.text)) - 1
            first_sent = nltk.sent_tokenize(article.text)[0]
            first_sent.replace("(CNN) -", '', 1)
            first_sent.replace("(CNN)-", '', 1)
            first_sent.replace("(CNN)", '', 1)
            second_sent = nltk.sent_tokenize(article.text)[1]
            second_sent.replace("(CNN) -", '', 1)
            second_sent.replace("(CNN)-", '', 1)
            second_sent.replace("(CNN)", '', 1)
            last_sent = nltk.sent_tokenize(article.text)[last_index]
            last_sent.replace("(CNN) -", '', 1)
            last_sent.replace("(CNN)-", '', 1)
            last_sent.replace("(CNN)", '', 1)

            sentences.append(first_sent)
            if first_sent != second_sent:
                sentences.append(second_sent)
            if second_sent != last_sent and first_sent != last_sent:
                sentences.append(last_sent)
            if first_sent == second_sent or first_sent==last_sent or second_sent==last_sent:
                print("There was the same sentence twice. It did not get duplicated.")

        except newspaper.article.ArticleException:
            print('didnt work for that url:', url)

    # Second bit, use the ~100 random sentences that I compiled to even the mix of random to climate related sentences
    extra_sentences = pd.read_excel("training_sentences.xlsx", sheet_name="Sheet3", index_col=0)
    for sentence in extra_sentences["random_sentence"]:
        sentence = sentence.replace('\xa0', ' ')
        sentences.append(sentence)

    ##################################
    # SAVE THE SENTENCES
    ##################################
    print("Saving Sentences...")
    np.save("random.npy", np.array(sentences))
    print("Completed saving sentences!")


    print("Number of random sentences:", len(sentences))
    print(75*'-')

# ==============================================================================#
# OLD CODE
# ==============================================================================#

# cnn_paper = newspaper.build('https://www.cnn.com')
#
# for article in cnn_paper.articles:
#     print(article.url)
#
# cnn_article_array = []
# for i in range(5):
#     cnn_article = cnn_paper.articles[i]
#     cnn_article.download()
#     cnn_article.parse()
#     cnn_article.nlp()
#     print(cnn_article.text)