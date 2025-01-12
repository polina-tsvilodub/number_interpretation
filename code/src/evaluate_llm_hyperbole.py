# evaluation script for morality
import csv
import os
import argparse

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from langchain import HuggingFacePipeline
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
from init_model import init_model
import pandas as pd

def parse_response(raw_response):
    if "a:" in raw_response.lower():
        if "a:" in raw_response:
            response = raw_response.split("a:")[1].lower().strip()
            return response
        elif "A:" in raw_response:
            response = raw_response.split("A:")[1].lower().strip()
            return response
        else:
            print(f"Response: {raw_response}")
            parsed_response = 1000
            return parsed_response
    elif "answer:" in raw_response.lower():
        response = raw_response.split("answer:")[1].lower().strip()
        return response
    else:
        print(f"Response: {raw_response}")
        parsed_response = 1000
        return parsed_response


parser = argparse.ArgumentParser()

# model args
parser.add_argument('--model', type=str, default='gpt-4o-mini', help='model name')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=10, help='max tokens')
parser.add_argument('--prompt', type=str, default="0shot", help='prompt')

# eval args
parser.add_argument('--num', '-n', type=int, default=1, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')

# data args (I need to set up the input and output directory of my data)
parser.add_argument('--data_dir', type=str, default='../../data/', help='data directory')
parser.add_argument('--output_dir', type=str, default='../../data/results_pt/', help='output directory')
parser.add_argument('--expt_num', type=str, default='1a', help='Number of experiment.')


# parse args
args = parser.parse_args()

# read data (data should just be a list)

if (args.expt_num == "1a") or (args.expt_num == "1b"):
    datafile = "experiment_1_full"
elif args.expt_num == "2":
    datafile = "experiment_2_full"
elif (args.expt_num == "3a") or (args.expt_num == "3b"):
    datafile = "experiment_3_full"
else:
    raise ValueError(f"Data for experiment number {args.expt_num} not found.")

data = pd.read_csv(os.path.join(args.data_dir, f"{datafile}.csv"))

# get prompt

PROMPT_DIR = "../prompt_instructions/"
if args.prompt == "0shot":
    if args.expt_num == "1a":
        with open(os.path.join(PROMPT_DIR, "evaluation_0shot_1a_v3.txt"), 'r') as f:
            prompt = f.read().strip()
    elif args.expt_num == "1b":
        with open(os.path.join(PROMPT_DIR, "evaluation_0shot_1b_v3.txt"), 'r') as f:
            prompt = f.read().strip()
    elif args.expt_num == "2":
        with open(os.path.join(PROMPT_DIR, "evaluation_0shot_2_v3.txt"), 'r') as f:
            prompt = f.read().strip()
    elif args.expt_num == "3a":
        with open(os.path.join(PROMPT_DIR, "evaluation_0shot_3a_v3.txt"), 'r') as f:
            prompt = f.read().strip()
    elif args.expt_num == "3b":
        with open(os.path.join(PROMPT_DIR, "evaluation_0shot_3b_v3.txt"), 'r') as f:
            prompt = f.read().strip()
    else:
        raise ValueError("Prompt for given experiment number not found.")
else:
    raise ValueError(f"Prompt {args.prompt} not found.")
    
    
# initialize LLM (Don't need to change)
# if args.model in ["gpt-4-0613", "gpt-3.5-turbo", "gpt-4o-mini"]:
    
llm = init_model(model_name=f"{args.model}",
                temperature=args.temperature,
                max_tokens = args.max_tokens)
# elif args.model in ["claude-2"]:
#     llm = ChatAnthropic(model_name=args.model,
#                     temperature=args.temperature,
#                     max_tokens = args.max_tokens)
# elif args.model in ["llama-2-7b-chat"]:
#     llm = HuggingFacePipeline.from_model_id(
#         model_id="meta-llama/Llama-2-7b-chat-hf",
#         task="text-generation",
#         model_kwargs={"temperature": args.temperature, "max_length": args.max_tokens},
#     )
# else:
#     raise ValueError(f"Model {args.model} not found.")
    
    
# evaluate (I need to replace the code below to let it compatible with hyperbole dataset)

# I should pay attention to the args.num as it controls the number of cases will be evaluated
for j in range(args.num):
    graded_answers = []
    data_out = data.copy()

    for i in tqdm(range(args.offset, len(data))):
        
        if args.expt_num == "1b":
            # query for 1b
            unique_prices = ["50", "51", "500", "501", "1000", "1001", "5000", "5001", "10000", "10001"]
            query_1b = data.loc[i, "context"] + ". " + data.loc[i, "question_prefix"] + "'" + data.loc[i, "question_affect"] + "' " + data.loc[i, "response_prefix"] + "'" + data.loc[i, "utterance_template"] + "' " + data.loc[i, "task_1b"]
            queries_1b = [query_1b + str(p) + "." for p in unique_prices]
            parsed_resps = []
            for q in queries_1b:
                if "llama" in args.model:
                    template = f"Instructions: {prompt}\n{query}\nA:"
                    response = llm(template)[0]
                else:
                    messages = [SystemMessage(content=prompt), HumanMessage(content=q)]
                    response = llm.generate([messages], stop=["Q:"]).generations[0][0].text
                parsed_resps.append(parse_response(response))
            parsed_response = ", ".join([str(s) for s in parsed_resps])
        else:
            if args.expt_num == "1a":
                query = data.loc[i, "context"] + ". " + data.loc[i, "question_prefix"] + "'" + data.loc[i, "question_affect"] + "' " + data.loc[i, "response_prefix"] + "'" + data.loc[i, "utterance_template"] + "' " + data.loc[i, "task_1a"]
            
            elif args.expt_num == "2":
                query = data.loc[i, "context"] + ". " + data.loc[i, "state_template"] + data.loc[i, "question_prefix"] + "'" + data.loc[i, "question_affect"] + "' " + data.loc[i, "response_prefix"] + "'" + data.loc[i, "utterance_template"] + "' " + data.loc[i, "task_2"]

            elif args.expt_num == "3a":
                query = data.loc[i, "context"] + ". " + data.loc[i, "state_template"] + data.loc[i, "task_3a"]

            elif args.expt_num == "3b":
                query = data.loc[i, "context"] + ". " + data.loc[i, "state_template"] + data.loc[i, "task_3b"]
            else:
                raise ValueError(f"Experiment number {args.expt_num} not found.")
            
            if "llama" in args.model:
                    template = f"Instructions: {prompt}\n{query}\nA:"
                    response = llm(template)[0]
            else:
                messages = [SystemMessage(content=prompt), HumanMessage(content=q)]
                response = llm.generate([messages], stop=["Q:"]).generations[0][0].text
            parsed_response = parse_response(response)
        # parse response
        if args.verbose:
            print("--------------------------------------------------")
            print(f"Prompt: {prompt}")
            print(f"Story: {query}")
            print(f"A: {response}")
            print(f"Parsed A: {parsed_response}")

        # append to list
        graded_answers.append(parsed_response)
    data_out["parsed_response"] = graded_answers
    # write to file
    if not os.path.exists(os.path.join(args.output_dir, datafile)):
        os.makedirs(os.path.join(args.output_dir, datafile))

    prefix = f"{args.model.replace('/','_')}_{args.prompt}_{args.temperature}_{args.num}_{args.offset}_iter{j}_rep"
    data_out.to_csv(os.path.join(args.output_dir, datafile, f"{prefix}_{args.expt_num}_predicted_answers.csv"), index=False)
