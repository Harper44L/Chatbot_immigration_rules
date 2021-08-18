from QAGenerator import QAGenerator
from QAGenerator import print_qa

qg = QAGenerator()
with open('immigration_rules.txt', 'r') as a:
    article = a.read()

qa_list = qg.generate(
    article,
    answer_style="detailed"
)
print(qa_list)

import csv
# 1. 创建文件对象
# f = open('qapairs.csv', 'w', encoding='utf-8')
f = open('qapairs_d.csv', 'w', encoding='utf-8')
# 2. 基于文件对象构建 csv写入对象
csv_writer = csv.writer(f)
# 3. 构建列表头
csv_writer.writerow(["Index","Question","Answer","Validity","Score"])
# 4. 写入csv文件内容
for pair in qa_list:
    if(pair["validity"]=='YES'):
        csv_writer.writerow([pair['index'],pair['question'],pair['answer'].replace('</s>',''),pair["validity"],pair['score']])
# 5. 关闭文件
f.close()

