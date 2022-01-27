#Import packages
import streamlit as st
import os
from math import*
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from nltk.corpus import stopwords
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from nltk.cluster.util import cosine_distance
import networkx as nx
#from gensim.summarization.summarizer import summarize
from bs4 import BeautifulSoup
import requests
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import string

#Need to add manually due to Heroku
sw = stopwords.words('english')

@st.cache
def generate_summary(article, top_n = 3):
    summarize_text = []
    sentences =  clean_sentences(article)
    #Find similar sentences
    sentence_sim_martix = sim_matrix(sentences)
    sentence_sim_graph = nx.from_numpy_array(sentence_sim_martix)
    scores = nx.pagerank(sentence_sim_graph)
    #Rank similarity and find summary sentences
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse = True)    
    for i in range(top_n):
        summarize_text.append(ranked_sentence[i][1])
    summary = " ".join(summarize_text)
    return summary
@st.cache
def generate_summary_jaccard(article, top_n = 3):
    summarize_text = []
    sentences =  clean_sentences(article)
    #Find similar sentences
    sentence_sim_martix = sim_matrix_j(sentences)
    sentence_sim_graph = nx.from_numpy_array(sentence_sim_martix)
    scores = nx.pagerank(sentence_sim_graph)
    #Rank similarity and find summary sentences
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse = True)    
    for i in range(top_n):
        summarize_text.append(ranked_sentence[i][1])
    summary = " ".join(summarize_text)
    return summary
@st.cache
def generate_summary_spacy(article, n_sent = 3):
    try:
        nlp = spacy.load("en_core_web_md")
    except: # If not present, we download
        spacy.cli.download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")
    text = nlp(article)
    tokens = [token.text for token in text]
    word_frequencies = {}
    for word in text:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
                    
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
        
    sentence_tokens= [sent for sent in text.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
                    
    summary = nlargest(n_sent, sentence_scores, key = sentence_scores.get)
    final_summary = [word.text for word in summary]
    summary = ''.join(final_summary)
    summary = summary.replace('\n','')
    summary = re.sub(r'(?<=[.,\?!:])(?=[^\s])', r' ', summary)
    return summary
@st.cache
def generate_summary_lsa(article):
    parser = PlaintextParser.from_string(article, Tokenizer('english'))
    lsa = LsaSummarizer()
    lsa_summary = lsa(parser.document, 3)
    summary = ''
    for s in lsa_summary: 
        if summary == '':
            summary = str(s)
        else:
            summary = summary + ' ' + str(s)
    return summary
@st.cache
def generate_summary_lexrank(article):
    parser = PlaintextParser.from_string(article, Tokenizer('english'))
    lex_rank = LexRankSummarizer() 
    lex_summary = lex_rank(parser.document, 3)
    summary = ''
    for s in lex_summary: 
        if summary == '':
            summary = str(s)
        else:
            summary = summary + ' ' + str(s)
    return summary
@st.cache
def generate_summary_genism(article, word_count = 250):
    sentences = clean_sentences(article)
    text = " ".join(sentences)
    summary = summarize(text, word_count = word_count)
    summary = summary.replace('\n',' ')
    return summary
@st.cache
def sent_sim(sent1, sent2, stopwords = None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
@st.cache
def sim_matrix(sent, stop_words = sw):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sent), len(sent)))
 
    for ind1 in range(len(sent)):
        for ind2 in range(len(sent)):
            if ind1 == ind2:
                continue 
            similarity_matrix[ind1][ind2] = sent_sim(sent[ind1], sent[ind2], stop_words)
    return similarity_matrix
@st.cache
def sent_sim_jaccard(sent1, sent2, stopwords = None):
    if stopwords is None:
        stopwords = []
        
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
        
    intersection = len(set.intersection(*[set(vector1), set(vector2)]))
    union = len(set.union(*[set(vector1), set(vector2)]))
    return intersection/float(union)
@st.cache
def sim_matrix_j(sent, stop_words = sw):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sent), len(sent)))
 
    for ind1 in range(len(sent)):
        for ind2 in range(len(sent)):
            if ind1 == ind2:
                continue 
            similarity_matrix[ind1][ind2] = sent_sim_jaccard(sent[ind1], sent[ind2], stop_words)
    return similarity_matrix
@st.cache
def clean_sentences(text):
    text = text.replace('\n','')
    #Get rid of links
    text = re.sub(r'www\.[a-z]?\.?(com)+|[a-z]+\.(com)', '', text)
    #Add space after punctuation if its not there
    text = re.sub(r'(?<=[.,\?!:])(?=[^\s])', r' ', text)
    text = text.lower()
    #Get rid of punctuation
    text.replace("[^a-zA-Z]", " ").split(" ")
    sent = sent_tokenize(text)
    return sent
@st.cache
def scrape_article(link):
    story_page = requests.get(link)
    story_soup = BeautifulSoup(story_page.text, 'html.parser')
    sections = story_soup.find_all('section')
    story_paragraphs = []
    for section in sections:
        paragraphs = section.find_all('p')
        for paragraph in paragraphs:
            story_paragraphs.append(paragraph.text)
    article_text = ''
    for p in story_paragraphs:
        if article_text == '':
            article_text = p
        else:
            article_text = article_text + ' ' + p
    return article_text


st.title('Summary Methods In Python')
st.subheader('An example of various extractive summary methods that can be done in Python to compare and contrast the results!')

#Make a form to enter the review
form = st.form(key = "text_form")
article = form.text_area(label="Enter the text of your article to summarize:")
submit = form.form_submit_button(label="Generate Summaries From Text")
word_cloud = form.checkbox(label = 'Visualize 20 Most Frequent Words')

link_form = st.form(key = "link_form")
link_article = link_form.text_input(label = "Enter the link to the medium article you want to summarize:")
link_submit = link_form.form_submit_button(label="Generate Summaries From Article")
word_cloud = link_form.checkbox(label = 'Visualize 20 Most Frequent Words')

if submit:
    #Make prediction from the input text
    cosine = generate_summary(article)
    jaccard = generate_summary_jaccard(article)
    #genism = generate_summary_genism(article)
    spacy_sum = generate_summary_spacy(article)
    lsa = generate_summary_lsa(article)
    lexrank = generate_summary_lexrank(article)

 
    #Display different summaries
    st.header("Results")
    st.write('**Cosine summary:** ')
    st.write(cosine)
    st.write('**Jaccard summary:** ')
    st.write(jaccard)
    st.write('**SpaCy summary:** ')
    st.write(spacy_sum)
    st.write('**LSA summary:** ')
    st.write(lsa)
    st.write('**LexRank summary:** ')
    st.write(lexrank)

    if word_cloud:
        #Instantiates a frequency dictionary
        fdist = FreqDist()
        #Counts frequencies of words in example text
        punct = list(string.punctuation)
        punct.append("’")
        punct.append("‘")
        punct.append("-")
        punct.append("“")
        punct.append("”")
        for word in word_tokenize(article):
            if (word in punct) or (word in sw):
                continue
            else:
                fdist[word.lower()] += 1
        #Look at distribution of frequencies of top 20 words
        fdist.plot(20, title = 'Frequency of Top 20 Words in Article')
        st.pyplot(plt)


if link_submit:
    if not bool(re.search(r'medium.com', link_article)):
        st.warning("Please make sure the link is to a Medium article")
    else:
        article = scrape_article(link_article)
        #Make prediction from the input text
        cosine = generate_summary(article)
        jaccard = generate_summary_jaccard(article)
        #genism = generate_summary_genism(article)
        spacy_sum = generate_summary_spacy(article)
        lsa = generate_summary_lsa(article)
        lexrank = generate_summary_lexrank(article)

    
        #Display different summaries
        st.header("Results")
        st.write('**Cosine summary:** ')
        st.write(cosine)
        st.write('**Jaccard summary:** ')
        st.write(jaccard)
        st.write('**SpaCy summary:** ')
        st.write(spacy_sum)
        st.write('**LSA summary:** ')
        st.write(lsa)
        st.write('**LexRank summary:** ')
        st.write(lexrank)

        if word_cloud:
            #Instantiates a frequency dictionary
            fdist = FreqDist()
            #Counts frequencies of words in example text
            punct = list(string.punctuation)
            punct.append("’")
            punct.append("‘")
            punct.append("-")
            punct.append("“")
            punct.append("”")
            for word in word_tokenize(article):
                if (word in punct) or (word in sw):
                    continue
                else:
                    fdist[word.lower()] += 1
            #Look at distribution of frequencies of top 20 words
            fdist.plot(20, title = 'Frequency of Top 20 Words in Article')
            st.pyplot(plt)
    