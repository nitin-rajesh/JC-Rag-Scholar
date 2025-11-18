import json
import ast
import re

jcp = []
with open('jc_model_answers.json') as fp:
    jcp = json.load(fp)



for obj in jcp:
    jobj = json.loads(obj['model_answer'])
    obj['contexts'] = jobj['context']
    obj['model_answer'] = jobj['answer']

with open('jc_ans_with_context.json',mode="w") as fp:
    json.dump(jcp, fp, indent=4)