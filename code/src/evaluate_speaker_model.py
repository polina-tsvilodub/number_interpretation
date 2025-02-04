"""
Script for evaluating LLM as the forward pragmatic speaker model.
Critical prediction: hyperbolic / round utterances more likely under affect and fuzzy goals, than non-affect and precise goals.
With this accurate model, we should be able to invert the speaker model to get the listener model (since the priors of the LM are fairly accurate).
"""
import pandas as pd
from init_model import init_model
import os
import argparse
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from tqdm import tqdm
import numpy as np

NUM_ITER = 1
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
parser.add_argument('--model', type=str, default='gpt-4o-mini', help='model name')
parser.add_argument('--temperature', type=float, default=1, help='temperature')
parser.add_argument('--max_tokens', type=int, default=512, help='max tokens')
parser.add_argument('--num_completions', type=int, default=1, help='number of completions')

# eval args
parser.add_argument('--num', '-n', type=int, default=300, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')

# data args (I need to set up the input and output directory of my data)
parser.add_argument('--data_dir', type=str, default='../../data/', help='data directory')
parser.add_argument('--output_dir', type=str, default='../../data/results_pt/', help='output directory')
parser.add_argument('--use_generation', type=bool, default=False, help='whether to use generation or scoring (default).')



# parse args
args = parser.parse_args()

filename = "experiment_1b_raw"
if args.model in ["gpt-4-0613", "gpt-3.5-turbo", "gpt-4o-mini", "gemini-1.5-pro", "claude-3-5-sonnet-20241022"]:
    llm = init_model(model_name=args.model,
                    temperature=args.temperature,
                    max_tokens = args.max_tokens)
else:
    raise NotImplementedError(f"Model {args.model} not implemented yet.")

affect_conditions = {
    '0': "{name} thinks that the price of the {item} is appropriate. ",
    '1': "{name} thinks the {item} is too expensive. "
}
goals = {
    "affect" : "{name} wants to communicate their attitude towards the price of the {item} they bought. ",
    "state": "{name} wants to communicate the price of the {item} they bought. ",
    "both": "{name} wants to communicate both the price of the {item} they bought and their attitude towards the price. "
}
halo = {
    "exact": "{name} wants to precisely communicate the price of the {item} they bought. ",
    "fuzzy": "{name} wants to communicate the approximate price of the {item} they bought. "
}
conditions = [
    ("state", "fuzzy", "0"),
    ("state", "exact", "0"),
    ("affect", "fuzzy", "0"),
    ("affect", "exact", "0"),
    ("both", "fuzzy", "0"),
    ("both", "exact", "0"),
    ("state", "fuzzy", "1"),
    ("state", "exact", "1"),
    ("affect", "fuzzy", "1"),
    ("affect", "exact", "1"),
    ("both", "fuzzy", "1"),
    ("both", "exact", "1")
]
question_template = "A friend asked {name} if the {item} was expensive. "
utterance_template = "How likely is it that {name} will say: 'The {item} cost {utterance}.'?"
utterance_production_template = "{name} says: 'The {item} cost $"
state_template = " The {item} cost {state}. "

if args.use_generation:
    with open(os.path.join("../prompt_instructions/advanced_prompting", "evaluation_0shot_1b_v2_speaker_free_production.txt"), 'r') as f:
        system_prompt = f.read().strip()
else:    
    with open(os.path.join("../prompt_instructions/advanced_prompting", "evaluation_0shot_1b_v2_speaker_scoring.txt"), 'r') as f:
        system_prompt = f.read().strip()


parsed_answers = []
goal_lists = []
affect_valences = []
halo_lists = []
utterances_lists = []
states_list = []
item_lists = []
name_lists = []
names = pd.read_csv("../../data/experiment_1_full.csv")["name"].unique()
print("names ", names)

items = ["electric kettle", "laptop", "watch"]
prices = ["$50", "$51", "$500", "$501", "$1000", "$1001", "$5000", "$5001", "$10000", "$10001"]

if not os.path.exists(os.path.join(args.output_dir, filename)):
        os.makedirs(os.path.join(args.output_dir, filename))

# iterate over the stories to model the pragmatic speaker
for iter in tqdm(range(NUM_ITER)):
    prefix = f"{args.model.replace('/','_')}_speaker_test_{args.temperature}_{args.num}_{args.offset}_iter{iter}"
    results_path =  os.path.join(args.output_dir, filename, f"{prefix}_predicted_answers_scoring_fixedPrompt.csv")
    results_path_final = os.path.join(args.output_dir, filename, f"{prefix}_predicted_answers_scoring_fixedPrompt_final.csv")
    for item in items:
        name = np.random.choice(names)
        if item == "electric kettle":
            init_prompt = f"{name} bought an {item}. "
        else:
            init_prompt = f"{name} bought a {item}. "

        for s in prices:
            prompt = init_prompt + f"The {item} cost {s}. "
            prompt += question_template.format(name=name, item=item)
            for c in conditions:
                # construct prompt based on goal
                if c[0] == "both":
                    goal_prompt = prompt + goals[c[0]].format(name=name, item=item) + halo[c[1]].format(name=name, item=item) + affect_conditions[c[2]].format(name=name, item=item)
                elif c[0] == "state":
                    goal_prompt = prompt + goals[c[0]].format(name=name, item=item) + halo[c[1]].format(name=name, item=item) 
                elif c[0] == "affect":
                    goal_prompt = prompt + goals[c[0]].format(name=name, item=item) + affect_conditions[c[2]].format(name=name, item=item)
                
                for u in prices:
                    
                    # construct prompt
                    full_prompt = goal_prompt + utterance_template.format(name=name, item=item, utterance=u)
                    print("------ full prompt------- ", full_prompt) 
                    print("system prompt", system_prompt)
                    # record
                    goal_lists.append(c[0])
                    halo_lists.append(c[1])
                    affect_valences.append(c[2])
                    # utterances_lists.append(r["utterance"])
                    item_lists.append(item)
                    name_lists.append(name)

                    states_list.append(s)
                    utterances_lists.append(u)

                    if args.model in ["gpt-4-0613", "gpt-3.5-turbo", "claude-2", "gpt-4o-mini", "gemini-1.5-pro", "claude-3-5-sonnet-20241022"]:
                        try:
                            messages = [SystemMessage(content=system_prompt), HumanMessage(content=full_prompt)]
                            response = llm.generate([messages], stop=["Q:"]).generations[0][0].text
                        except:
                            response = "API error"
                    elif args.model in ["llama-2-7b-chat"]:
                        template = f"Instructions: {system_prompt}\n{full_prompt}\nA:"
                        response = llm(template)[0]
                    

                    if args.verbose:
                        print("--------------------------------------------------")
                        print(f"Instruction: {system_prompt}")
                        print(f"Story: {prompt}")
                        print(f"A: {response}")
                        

                    # append to list
                    parsed_answers.append(response)
            
                    results = pd.DataFrame({
                        "halo": c[1],
                        "goal": c[0],
                        "affect_valence": c[2],
                        "name": name,
                        "utterance": u,
                        "state": s,
                        "item": item,
                        "parsed_answer": response,
                    }, index=[iter])
                    # continuous writing to file
                    results.to_csv(
                        results_path,
                        index=False,
                        mode="a",
                        header=not os.path.exists(
                            results_path
                        )
                    )

    df_out = pd.DataFrame({
        "halo": halo_lists,
        "goal": goal_lists,
        "affect_valence": affect_valences,
        "name": name_lists,
        "utterance": utterances_lists,
        "state": states_list,
        "item": item_lists,
        "parsed_answer": parsed_answers,
    })
    
    df_out.to_csv(
        results_path_final,
        index=False
    )

