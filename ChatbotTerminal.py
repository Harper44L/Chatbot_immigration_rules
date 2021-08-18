from PreProcessor import PreProcessor
import csv
from sklearn.metrics.pairwise import cosine_similarity
import re, math

from nltk.translate.bleu_score import sentence_bleu


csv_reader = csv.reader(open("qapairs.csv"))

pre_processor = PreProcessor()


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator
from nltk.tokenize import word_tokenize
def chatbot_reply(query):
    query = pre_processor.pre_process_qa(query)
    query_v = pre_processor.text_to_vector(query)
    max_cos = 0
    max_index = 0
    csv_reader = csv.reader(open("qapairs_d.csv"))
    for qapair in csv_reader:
        question = pre_processor.pre_process_qa(qapair[1])
        question_v = pre_processor.text_to_vector(question)
        question = qapair[1]
        cos = get_cosine(question_v, query_v)

        # question_s=word_tokenize(question)
        # query_s=word_tokenize(query)
        #
        # cos=sentence_bleu(query_s,question_s)

        if (cos > max_cos):
            max_cos = cos
            max_index = qapair[0]

    csv_reader = csv.reader(open("qapairs_d.csv"))
    for qapair in csv_reader:
        if qapair[0] == max_index:
            return qapair[2]


Flag = True
while (Flag):
    query = input("Please input your query >>>")
    print("ROBO: ", end="")
    print(chatbot_reply(query))


