import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, jsonify, render_template, request


app = Flask(__name__, template_folder='./')


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    response =  request.args.get('message')
    return response

@app.route('/')
def main():
    return render_template('myindex.html')


if __name__ == '__main__':
    app.run(debug=True)