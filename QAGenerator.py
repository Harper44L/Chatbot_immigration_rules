import numpy as np
import torch
import spacy
import en_core_web_sm
from transformers import (
    AutoTokenizer,  # 分词器
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
from PreProcessor import PreProcessor
from QAEvaluator import QAEvaluator


class QAGenerator:
    def __init__(self, model_dir=None):

        QG_PRETRAINED_Model = "iarfmoose/t5-base-question-generator"
        # t5 Text-To-Text-Transfer-Transformer的简称，是Google在2019年提出的一个新的NLP模型。
        self.ANSWER_TOKEN = "<answer>"
        self.CONTEXT_TOKEN = "<context>"
        self.SEQ_LENGTH = 512

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = PreProcessor();
        self.tokenizer = AutoTokenizer.from_pretrained(QG_PRETRAINED_Model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(QG_PRETRAINED_Model)
        self.model.to(self.device)
        self.evaluator = QAEvaluator(model_dir)  # model_dir :The folder that the trained model checkpoints are in

    def generate(self, article, answer_style="all",evaluate=True):
        print("start to generate questions...\n")
        qg_inputs, qg_answers = self.generate_inputs_answers(article, answer_style)
        generated_questions = self.generate_questions_from_inputs(qg_inputs)  # generate questions from inputs
        qa_list=[]

        if evaluate == True:
            print("start to evaluate QA pairs...\n")
            encoded_qa_pairs = self.evaluator.encode_qa_pairs(
                generated_questions, qg_answers
            )
            scores = self.evaluator.evaluate_qa_pairs(encoded_qa_pairs)
            # qa_list = self.get_ranked_qa_pairs(
            #     generated_questions, qg_answers, scores
            # )
            qa_list=self.get_qa_pairs(
                generated_questions, qg_answers, scores
            )
        return qa_list

    def generate_inputs_answers(self, text, answer_style):

        paragraphs = self.preprocessor.split_into_paragraphs(text)
        # paragraphs = self.preprocessor.clean_punctuation(paragraphs)

        VALID_ANSWER_STYLES = ["all", "detailed", "concise"]
        inputs = []
        if answer_style == "detailed" or answer_style == "all":
            answers=[]
            for paragraph in paragraphs:
                title=paragraph.split(":")[0]
                # inputs_for_generate_question, answer = self.generate_input_answer_detailed(
                #     title, paragraph
                # )
                answer = paragraph.split(":")[1]
                input_for_question = "{} {} {} {}".format(
                    self.ANSWER_TOKEN, title, self.CONTEXT_TOKEN, answer
                )

                inputs.append(input_for_question)
                answers.append(answer)

        if answer_style == "concise" or answer_style == "all":
            inputs_for_generate_questions, answers = self.generate_inputs_answers_concise(paragraphs)

            inputs.extend(inputs_for_generate_questions)
            # answers.extend(answers)
        return inputs, answers

    def generate_inputs_answers_concise(self, paragraphs):
        spacy_nlp = en_core_web_sm.load()
        docs = list(spacy_nlp.pipe(paragraphs, disable=["parser"]))
        inputs_for_question = []
        answers = []
        for i in range(len(paragraphs)):
            entities = docs[i].ents
            if entities:
                for entity in entities:
                    print(entity)
                    input = "{} {} {} {}".format(
                        self.ANSWER_TOKEN, entity, self.CONTEXT_TOKEN, paragraphs[i].replace('</s>', '')
                    )
                    answers.append(entity)
                    inputs_for_question.append(input)
        return inputs_for_question, answers

    def generate_questions_from_inputs(self, inputs):
        generated_questions = []
        for input in inputs:
            self.model.eval()
            encoded_input = self.tokenizer(
                input,
                padding='max_length',
                max_length=self.SEQ_LENGTH,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.model.generate(input_ids=encoded_input["input_ids"])
            question = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # question = self._generate_question(input)
            generated_questions.append(question)
        return generated_questions

    def get_ranked_qa_pairs(
            self, generated_questions, qg_answers, scores
    ):

        qa_list = []
        for i in range(len(scores)):
            print("***********************")
            print(scores)
            index = scores[i]
            print(index)

            qa = {}
            # qa["question"] = generated_questions[index].split("?")[0]
            qa["index"]=i+1
            qa["question"]=generated_questions[index].split("?")[0] + "?"
            qa["answer"] = qg_answers[index]
            qa["score"] = scores[i]
            print(generated_questions[index].split("?")[0] + "?")
            print(qg_answers[index])


            qa_list.append(qa)
        return qa_list

    def get_qa_pairs(
            self, generated_questions, qg_answers, scores
    ):

        qa_list = []
        validity_count=0
        for i in range(len(generated_questions)):
            index = i
            qa = {}
            # qa["question"] = generated_questions[index].split("?")[0]
            qa["index"]=i+1
            qa["question"]=generated_questions[index].split("?")[0] + "?"
            qa["answer"] = qg_answers[index]
            qa["score"] = scores[i]
            if(scores[i]>0):
                qa["validity"]= "YES"
                validity_count+=1
            else:
                qa["validity"] = "NO"

            qa_list.append(qa)
        validity_percentage=validity_count/len(generated_questions)
        print("validity percentage over qa pairs is :{}".format(validity_percentage))
        return qa_list


def print_qa(qa_list, show_answers=True):
    for i in range(len(qa_list)):
        print("**"*50)
        print(" ")
        space = " " * int(np.where(i < 9, 3, 4))  # wider space for 2 digit q nums
        print("{}) Q: {}".format(i + 1, qa_list[i]["question"]))
        answer = qa_list[i]["answer"]
        if show_answers:
            print("{}A:".format(space), answer, "\n")
        print("The validity is : {}".format(qa_list[i]["validity"]))
        print("The score is : {}".format(qa_list[i]["score"]))
        print(" ")