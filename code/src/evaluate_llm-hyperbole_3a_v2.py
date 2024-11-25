# evaluation script for morality
import csv
import os
import argparse

from tqdm import tqdm
from crfm import crfmChatLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from init_model import init_model

# parse raw response
def parse_response(raw_response):
    if "a:" in raw_response.lower():
        if "a:" in raw_response:
            response = raw_response.split("a:")[1].lower().strip()
        elif "A:" in raw_response:
            response = raw_response.split("A:")[1].lower().strip()
        else:
            print(f"Response: {raw_response}")
            parsed_response = int(input("Enter response(1-5):"))
            return parsed_response
    elif "answer:" in raw_response.lower():
        response = raw_response.split("Answer:")[1].lower().strip()
    else:
        print(f"Response: {raw_response}")
        parsed_response = int(input("Enter response(1-5):"))
        return parsed_response

    if "impossible" in response or "1" in response:
        parsed_response = 1
    elif "not very likely" in response or "2" in response:
        parsed_response = 2
    elif "neutral" in response or "3" in response:
        parsed_response = 3
    elif "very likely" in response or "4" in response:
        parsed_response = 4
    elif "extremely likely" in response or "5" in response:
        parsed_response = 5
    else:
        print(f"Response {raw_response} not found.")
        parsed_response = int(input("Enter response(1-5):"))
    return parsed_response

parser = argparse.ArgumentParser()

# model args
parser.add_argument('--model', type=str, default='gpt-4-0613', help='model name')
parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=10, help='max tokens')
parser.add_argument('--prompt', type=str, default="0shot", help='prompt')

# eval args
parser.add_argument('--num', '-n', type=int, default=50, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')

# data args (I need to set up the input and output directory of my data)
parser.add_argument('--data_dir', type=str, default='../../data/', help='data directory')
parser.add_argument('--output_dir', type=str, default='../../data/results_pt/', help='output directory')


# parse args
args = parser.parse_args()


# no condiction is needed or I could change it to ["electric kettle", "watch", "laptop"] and construct the code in a different way

# read data (data should just be a list)

datafile = "experiment_3a_v2"
data = []
with open(os.path.join(args.data_dir, f"{datafile}.csv"), 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        story = ' '.join(row)
        data.append(story)


# get prompt

PROMPT_DIR = "../prompt_instructions/"
if args.prompt == "0shot":
    with open(os.path.join(PROMPT_DIR, "evaluation_0shot_3a_v2.txt"), 'r') as f:
        prompt = f.read().strip()
# elif args.prompt == "0shot_cot":
#     with open(os.path.join(PROMPT_DIR, "evaluation_0shot_cot_3a.txt"), 'r') as f:
#         prompt = f.read().strip()
else:
    raise ValueError(f"Prompt {args.prompt} not found.")
    
print("---- data passed to the model: ", len(data), data)

# initialize LLM (Don't need to change)
if args.model in ["gpt-4-0613", "gpt-3.5-turbo"]:
    # llm = ChatOpenAI(model_name=args.model,
    #                 temperature=args.temperature,
    #                 max_tokens = args.max_tokens)
    print("Model name ", args.model)
    llm = init_model(model_name=args.model,
                    temperature=args.temperature,
                    max_tokens = args.max_tokens)
elif args.model in ["claude-2"]:
    llm = ChatAnthropic(model_name=args.model,
                    temperature=args.temperature,
                    max_tokens = args.max_tokens)
elif args.model in ["llama-2-7b-chat"]:
    llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        task="text-generation",
        model_kwargs={"temperature": args.temperature, "max_length": args.max_tokens},
    )
else:
    raise ValueError(f"Model {args.model} not found.")
    
# evaluate (I need to replace the code below to let it compatible with hyperbole dataset)


# no condiction is needed or I could change it to ["electric kettle", "watch", "laptop"]

predicted_answers= []
graded_answers = []
# I should pay attention to the args.num as it controls the number of cases will be evaluated
for i in tqdm(range(args.offset, len(data))):
    story = data[i]
    query = story
    if args.model in ["gpt-4-0613", "gpt-3.5-turbo", "claude-2"]:
        messages = [SystemMessage(content=prompt), HumanMessage(content=query)]
        response = llm.generate([messages], stop=["Q:"]).generations[0][0].text
        print("Response: ", response)
    elif args.model in ["llama-2-7b-chat"]:
        template = f"Instructions: {prompt}\n{query}\nA:"
        response = llm(template)[0]

    # parse response
    parsed_response = parse_response(response)

    if args.verbose:
        print("--------------------------------------------------")
        print(f"Prompt: {prompt}")
        print(f"Story: {query}")
        print(f"A: {response}")
        print(f"Parsed A: {parsed_response}")

    # append to list
    predicted_answers.append(response)
    graded_answers.append(parsed_response)

# write to file
if not os.path.exists(os.path.join(args.output_dir, datafile)):
    os.makedirs(os.path.join(args.output_dir, datafile))

prefix = f"{args.model.replace('/','_')}_{args.prompt}_{args.temperature}_{args.num}_{args.offset}"
with open(os.path.join(args.output_dir, datafile, f"{prefix}_predicted_answers.txt"), 'w') as f:
    f.write('\n'.join(predicted_answers))
with open(os.path.join(args.output_dir, datafile, f"{prefix}_graded_answers.txt"), 'w') as f:
    f.write('\n'.join([str(x) for x in graded_answers]))