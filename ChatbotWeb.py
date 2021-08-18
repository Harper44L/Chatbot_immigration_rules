from flask import Flask, jsonify, render_template, request
import csv
import warnings
import math
from PreProcessor import PreProcessor

warnings.filterwarnings('ignore')
csv_reader = csv.reader(open("qapairs_d.csv"))

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

        if (cos > max_cos):
            max_cos = cos
            max_index = qapair[0]

    csv_reader = csv.reader(open("qapairs_d.csv"))
    for qapair in csv_reader:
        if qapair[0] == max_index:
            print(qapair[2])
            return qapair[2]
        # else:
        #     return "please enter again."

# webapp
app = Flask(__name__, static_folder='web', static_url_path='')


# 首先引入了Flask包，并创建一个Web应用的实例”app

# 定义路由规则, @app.route('/') 这个函数级别的注解指明了当地址是根路径时，就调用下面的函数。说的高大上些，这里就是MVC中的Contoller
# 当请求的地址符合路由规则时，就会进入该函数。
# 可以说，这里是MVC的Model层。你可以在里面获取请求的request对象，返回的内容就是response。
@app.route('/api/chat', methods=['GET'])
# @app.route('/chat', methods=['GET'])
# @app.route('/')
def chat():
    global query
    message = request.args.get('message')

    query = message
    resp = chatbot_reply(query)
    return resp

@app.route('/')
def main():
    return render_template('index.html')
    # return 'Hello World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# 当本文件为程序入口（也就是用python命令直接执行本文件）时，就会通过app.run()启动Web服务器。
# 如果不是程序入口，那么该文件就是一个模块。Web服务器会默认监听本地的5000端口，但不支持远程访问。
# 如果你想支持远程，需要在run()方法传入host=0.0.0.0，
