import datetime
import json
import os
from collections import defaultdict
from openai import OpenAI
from loguru import logger as eval_logger
from copy import deepcopy
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from tqdm import tqdm
from PIL import Image
dir_name = os.path.dirname(os.path.abspath(__file__))

REASONING_GRADING_PREFIX = """
You will be given a question, an ground truth answer and a model response. You need to extract the final answer from the model response, compare it with the ground truth answer, and then assign a binary score. Avoid providing explanations in your response. If there is no provided model response, please leave the extracted answer empty and give a score of 0. 

Your response must follow json formats with keys [extract_answer, score] where the value of the score is an interger in [0, 1]. You must follow the scoring rules:\n"""

REASONING_GRADING_INST = {
    1: \
    """
    ### Rules ###
    * Give a score of 1 if and only if the final answer and the ground truth answer are referring to the same term. It's acceptable to have different grammar or form (e.g., α and alpha; $R^2_{t,h,v,m}$ and R^2_t,h,v,m). It's also acceptable to have different orders of the terms when question asks for multiple terms.
    * Give a score of 0 if any term (e.g., ACC+ and ACC; P-101 and P=101) is different between the final answer and the ground truth.

    ### Example 1 Starts ###
    * Question: What is the name of the curve that intersects y=\lambda exactly three times?
    * Ground Truth: P56962
    * Response: There is only one curve that intersects y=\lambda exactly three times. The name of the curve is written as P55762.
    
    {
        "extracted_answer": "P55762",
        "score": 0
    }
    ### Example 1 Ends ###


    ### Example 2 Starts ###
    * Question: What is the letter of the subplot where all bars are above 35?
    * Ground Truth: (b)
    * Response: The letter of the subplot where all bars are above 35 is b.

    {
        "extracted_answer": "b",
        "score": 1
    }
    ### Example 2 Ends ###

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>

    """,

    2: \
    """
    ### Rules ###
    * If there are predefined options in the question:
        * Give a score of 1 if the final answer matches the ground truth answer exactly.
        * Give a score of 0 if the final answer does not match the ground truth answer.
    * If there are no predefined options in the question:
        * Give a score of 1 if the final answer shares the same semantic meaning with the ground truth answer (e.g., "increasing then decreasing" and "moving up then down"; "converge" and "move closer together").
        * Give a score of 0 if the final answer shares different semantic meanings from the ground truth answer (e.g., "increasing then decreasing" and "remain constant"; "converge" and "diverge").

    ### Example 1 Starts ###
    * Question: What is the trend of the red curve between t=10 and t=25?
    * Ground Truth: increasing then decreasing
    * Response: The red curve is increasing between t=10 and t=25.

    {
        "extracted_answer": "increasing",
        "score": 0
    }
    ### Example 1 Ends ###

    ### Example 2 Starts ###
    * Question: What is the interval where the blue curve achieves the maximum value among [0, 50], [50, 100], [100, 150], and [150, 200]?
    * Ground Truth: [50, 100]
    * Response: The interval where the blue curve achieves the maximum value is [50, 100].

    {
        "extracted_answer": "[50, 100]",
        "score": 1
    }
    ### Example 2 Ends ###

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>

    """,

    3: \
    """
    ### Rules ###
    * Give a score of 1 if and only if the two numbers are exactly equal in values. It's acceptable to have different notations (e.g., 0.01 and 10^-2; 1500 and 1.5e3).
    * Give a score of 0 if the two numbers are different in values.

    ### Example 1 Starts ###
    * Question: What is the value of the red curve at t=10?
    * Ground Truth: 0.01
    * Response: The value of the red curve at t=10 is 0.012.

    {
        "extracted_answer": "0.012",
        "score": 0
    }
    ### Example 1 Ends ###

    ### Example 2 Starts ###
    * Question: What is the value of the blue curve at t=50?
    * Ground Truth: 1500
    * Response: The value of the blue curve at t=50 is 1.5e3.

    {
        "extracted_answer": "1.5e3",
        "score": 1
    }
    ### Example 2 Ends ###

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>

    """,

    4: \
    """
    ### Rules ###
    * Give a score of 1 if and only if the two numbers are exactly equal in values. It's acceptable to have different notations (e.g., 0.01 and 10^-2; 1500 and 1.5e3).
    * Give a score of 0 if the two numbers are different in values.

    ### Example 1 Starts ###
    * Question: What is the value of the red curve at t=10?
    * Ground Truth: 0.01
    * Response: The value of the red curve at t=10 is 0.012.

    {
        "extracted_answer": "0.012",
        "score": 0
    }
    ### Example 1 Ends ###

    ### Example 2 Starts ###
    * Question: What is the value of the blue curve at t=50?
    * Ground Truth: 1500
    * Response: The value of the blue curve at t=50 is 1.5e3.

    {
        "extracted_answer": "1.5e3",
        "score": 1
    }
    ### Example 2 Ends ###

    ### Your Turn ###
    * Question: <|question|>
    * Ground Truth: <|ground_truth|>
    * Response: <|response|>

    """,
}




def math_doc_to_visual(doc):
    image_path = doc["image_url"]
    image = Image.open(image_path).convert("RGB")
    return [image]


def math_doc_to_text(doc, lmms_eval_specific_kwargs=None):

    prompt_template = """
    Answer the question.Do not give any explanation:
    question: {}
    """

    question = doc['query']
    options = doc['choice']
    
    prompt = prompt_template.format(question)
    if options!='NULL':
        options = eval(options)
        prompt += '\nchoices:\n'
        for i, choice in enumerate(options):
            prompt += f'{chr(65 + i)}:{choice}\n'
    print(prompt)
    return prompt



def math_process_results(doc, results):
    return {"math_acc_score": {"question_id": doc["id"], "question":doc['query'], "answer": doc["answer"],"pred":results[0]}}


def math_acc_results(results):
    query_list = []

    for result in results:
        # print(result['pred'])
        grading_query = REASONING_GRADING_PREFIX + deepcopy(\
            REASONING_GRADING_INST[1])\
            .replace("<|question|>", result['question'])\
            .replace("<|ground_truth|>", result['answer'])\
            .replace("<|response|>", result['pred'])
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": grading_query}
        ]
        query_list.append(messages)
    client = OpenAI(api_key='sk-s0zdpcVwUaHnBI5r949fC6D54aCd496494D2Ed17DfEcE290',base_url='https://yanlp.zeabur.app/v1')
    reps = []
    for q in tqdm(query_list):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=q,
            temperature=1.0,
            max_tokens=100,
            top_p=1.0,
            response_format={ "type": "json_object" }
        )
        reps.append(json.loads(response.choices[0].message.content))
    output = {}
    assert len(reps)==len(results)
    for i in range(0,len(reps)):
        rid = results[i]['question_id']
        pred = results[i]['pred']
        item = reps[i]
        item['pred'] = pred
        output[rid] = item
    with open('math.json', 'w') as file:
        json.dump(output, file, indent=4)
    return 666
