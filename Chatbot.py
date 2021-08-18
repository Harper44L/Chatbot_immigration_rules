
from QAGenerator import QAGenerator
from QAGenerator import print_qa

qg = QAGenerator()
with open('immigration_rules.txt', 'r') as a:
    article = a.read()

qa_list = qg.generate(
    article,
    answer_style="detailed"
)
# def bleu_score(sentence, gold_sentence):
#   return nltk.translate.bleu_score.sentence_bleu(
#       [gold_sentence], sentence)
print_qa(qa_list)

# f = open('qapairs.txt', 'w')
# for i in range(len(qa_list)):
#     print(qa_list[i]["question"])
#     print(qa_list[i]["answer"])
#     # space = " " * int(np.where(i < 9, 3, 4))  # wider space for 2 digit q nums
#     f.write(qa_list[i]["question"])
#     f.write("\n")
#     f.write(qa_list[i]["answer"])
#     f.write("\n")
#     f.write("\n")
# f.close



