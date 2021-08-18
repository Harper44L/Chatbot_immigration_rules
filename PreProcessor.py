from transformers import (
    AutoTokenizer,  # 分词器
    # AutoModelForSeq2SeqLM,
    # AutoModelForSequenceClassification,
)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re, math
from collections import Counter

WORD = re.compile(r'\w+')

class PreProcessor:

    def __init__(self, model_dir=None):

        QG_PRETRAINED_Model = "iarfmoose/t5-base-question-generator"
        self.tokenizer = AutoTokenizer.from_pretrained(QG_PRETRAINED_Model)


    def split_into_paragraphs(self, text):
        # MAX_TOKENS = 490
        paragraphs = text.split("\n")
        # tokenized_paragraphs = [
        #     self.tokenizer(sentence)["input_ids"] for sentence in paragraphs if len(sentence) > 0
        #
        # ]
        # paragraphs = []
        # while len(tokenized_paragraphs) > 0:
        #     paragraph = tokenized_paragraphs.pop(0)
        #     paragraphs.append(paragraph)
        for sentence in paragraphs:
            print(sentence)
            sentence=sentence.replace('</s>','')

        # return [self.tokenizer.decode(s) for s in paragraphs]
        return paragraphs
        # re_sentence = TreebankWordDetokenizer().detokenize(tokens)
        # return [self.tokenizer.decode(s) for s in paragraphs]

    def clean_punctuation(self, paragraphs):
        processed_paragraphs = []
        for sentence in paragraphs:
            if len(sentence) > 0:
                sentence = sentence.lower()  # 转成小写
                sentence = re.sub(r"[%s]+" % punctuation, "", sentence)
                print(sentence)
                processed_paragraphs.append(sentence)
        return processed_paragraphs

    def pre_process_qa(self, sentence):
        if len(sentence) > 0:
            sentence = sentence.lower()  # 转成小写
            sentence = self.clean_punctuation_qa(sentence)  # 去除标点符号
            tokens = word_tokenize(sentence)  # tokenize
            tokens = self.clean_stopwords(tokens)  # 去停用词
            tokens = self.lemmatization(tokens)
            # sentence=stemming(sentence) # 词干提取 用不用呢
            re_sentence = TreebankWordDetokenizer().detokenize(tokens)  # 恢复成句子

        return re_sentence

    def lemmatization(self, tokens):
        lemmatizer = WordNetLemmatizer()
        lemma_sentence = []
        for token in tokens:
            token = lemmatizer.lemmatize(token)
            lemma_sentence.append(token)
        return lemma_sentence


    def stemming(self, tokens):
        porter_stemmer = PorterStemmer()
        stemmed_sentence = []
        for token in tokens:
            token = porter_stemmer.stem(token)
            stemmed_sentence.append(token)
        return stemmed_sentence

    # 去停用词
    def clean_stopwords(self, tokens):
        stwords = stopwords.words('english')
        for token in tokens:
            if token in stwords:
                tokens.remove(token)  # 去除停用词
        return tokens

    def clean_punctuation_qa(self, sentence):
        sentence = re.sub(r"[%s]+" % punctuation, "", sentence)
        return sentence

    def text_to_vector(self, text):
        words = WORD.findall(text)
        return Counter(words)

def stemming(tokens):
    porter_stemmer = PorterStemmer()
    stemmed_sentence = []
    for token in tokens:
        token = porter_stemmer.stem(token)
        stemmed_sentence.append(token)
    return stemmed_sentence

