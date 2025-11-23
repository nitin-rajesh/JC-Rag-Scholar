import json
import ast
import re

jcp = []
with open('jc_model_answers.json', mode='r') as fp:
    jcp = json.load(fp)


for obj in jcp:
    answer = obj['model_answer']['answer']
    context = obj['model_answer']['context']

    raga_ctx = []
    for snippet in re.split(r"(\\u|\\n)\w+",context):
        if len(snippet) > len('SOURCE: act-3_scene-2.txt') and "Keywords" not in snippet:
            raga_ctx.append(snippet)
            print('->',snippet,end="<-\n")

    obj['model_answer'] = answer
    obj['contexts'] = raga_ctx

with open('jc_raga_ip.json',mode="w") as fp:
    json.dump(jcp, fp, indent=4)