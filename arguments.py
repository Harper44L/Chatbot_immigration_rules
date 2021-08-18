import argparse # 用于解析命令行参数和选项的标准模块
from Practice.QA_Generator import QA_Generator
from Practice.QA_Generator import print_qa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_dir",  # 字符串的名字或者列表
        default='articles/Immigration.txt',
        type=str,
        required=True,  # 可选参数是否可以省略 (仅针对可选参数)
        help="The text that will be used as context for question generation.",  # 将被用作问题生成上下文的文本
    )

    # 给parser实例增加一个'--text_dir'属性
    parser.add_argument(
        "--model_dir",
        default=None,
        type=str,
        help="The folder that the trained model checkpoints are in.",
    )
    parser.add_argument(
        "--num_questions",
        default=10,
        type=int,
        help="The desired number of questions to generate.",
    )

    parser.add_argument(
        "--show_answers",
        default='True',
        type=parse_bool_string,
        help="Whether or not you want the answers to be visible. Choose from ['True', 'False']",
    )
    parser.add_argument(
        "--use_qa_eval",
        default='True',
        type=parse_bool_string,
        help="Whether or not you want the generated questions to be filtered for quality. Choose from ['True', 'False']",
    )
    args = parser.parse_args()
    # 把parser中设置的所有"add_argument"给返回到args子类实例当中

    with open(args.text_dir, 'r') as file:
        text_file = file.read()

    qg = QA_Generator(args.model_dir)

    qa_list = qg.generate(
        text_file,
        num_questions=int(args.num_questions),
        answer_style=args.answer_style,
        use_evaluator=args.use_qa_eval
    )
    print_qa(qa_list, show_answers=args.show_answers)

# taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def parse_bool_string(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    main()
