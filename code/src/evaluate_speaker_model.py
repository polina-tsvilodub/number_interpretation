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
    # TODO: this should just grab the last line and extract the number, and there should be more new tokens allowed
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
parser.add_argument('--temperature', type=float, default=0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=512, help='max tokens')
parser.add_argument('--num_completions', type=int, default=1, help='number of completions')

# eval args
parser.add_argument('--num', '-n', type=int, default=300, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')

# data args (I need to set up the input and output directory of my data)
parser.add_argument('--data_dir', type=str, default='../../data/', help='data directory')
parser.add_argument('--output_dir', type=str, default='../../data/results_pt/', help='output directory')


# parse args
args = parser.parse_args()

filename = "experiment_1b_raw"
data_path = os.path.join(args.data_dir, f"{filename}.csv")
data = pd.read_csv(data_path)
# aproach 1: GPT-4o-mini rating results
if args.model in ["gpt-4-0613", "gpt-3.5-turbo", "gpt-4o-mini"]:
    # llm = ChatOpenAI(model_name=args.model,
    #                 temperature=args.temperature,
    #                 max_tokens = args.max_tokens)
    llm = init_model(model_name=args.model,
                    temperature=args.temperature,
                    max_tokens = args.max_tokens)
else:
    raise NotImplementedError(f"Model {args.model} not implemented yet.")

affect_conditions = {
    0: " Specifically, {name} thinks that the price of the {item} is appropriate. ",
    1: " Specifically, {name} thinks the {item} is too expensive. "
}
goals = {
    "a" : "{name} wants to communicate their attitude towards the price of the {item} they bought. ",
    "s_exact": "{name} wants to precisely communicate the price of the {item} they bought. ",
    "s_fuzzy": "{name} wants to roughly communicate the price of the {item} they bought. "
}
conditions = [
    "a-only-0", # exact
    "a-only-1",
    # "a-only-0", # FUZZY
    # "a-only-1",
    "only-s_exact-0",
    "only-s_exact-1",
    "only-s_fuzzy-0",
    "only-s_fuzzy-1",
    "a-s_exact-0",
    "a-s_exact-1",
    "a-s_fuzzy-0",
    "a-s_fuzzy-1",
]
question_template = "If a friend asked {name} if the {item} was expensive, how likely is it that {name} will say: 'The {item} cost {utterance}'?"
state_template = " The {item} cost {state}. "

with open(os.path.join("../prompt_instructions/", "evaluation_0shot_1b_v2_speaker_free_production.txt"), 'r') as f:
    system_prompt = f.read().strip()

predicted_answers = []
parsed_answers = []
affect_lists = []
affect_valences = []
goal_lists = []
utterances_lists = []
states_list = []
item_lists = []

# iterate over the stories to model the pragmatic speaker
for iter in tqdm(range(NUM_ITER)):
    for c in conditions:
        split_condition = c.split("-")
        # construct prompt based on goal
        if split_condition[0] == "a":
            affect_goal_prompt = goals["a"]
            affect_prompt = affect_conditions[int(split_condition[2])]
        else:
            affect_goal_prompt = ""
            affect_prompt = ""

        if split_condition[1] != "only":
            goal_prompt = goals[split_condition[1]]
        else:
            goal_prompt = ""
        

        for i, r in data.iterrows():
            name = r["context"].split(" ")[0]
            item = r["context"].split(" ")[-1]
            if item == "kettle":
                item = "electric kettle"
            # construct prompt
            # iterate over all states for full decomposition
            goal_p = goal_prompt.format(name=name, item=item) if goal_prompt != "" else ""
            affect_goal_p = affect_goal_prompt.format(name=name, item=item) if affect_goal_prompt != "" else ""
            free_production_template = f" A friend asks {name}: 'Was it expensive?' {name} responds: 'The {item} cost $"
            prompt = goal_p + affect_goal_p + affect_prompt.format(name=name, item=item) + state_template.format(item=item, state=r["state"]) + free_production_template #question_template.format(name=name, item=item, utterance=r["utterance"])
            # record
            affect_lists.append(c.split("-")[0])
            goal_lists.append(c.split("-")[1])
            affect_valences.append(c.split("-")[2])
            utterances_lists.append(r["utterance"])
            item_lists.append(item)
            states_list.append(r["state"])

            if args.model in ["gpt-4-0613", "gpt-3.5-turbo", "claude-2", "gpt-4o-mini"]:
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
                response = llm.generate([messages], stop=["Q:"]).generations[0][0].text
            elif args.model in ["llama-2-7b-chat"]:
                template = f"Instructions: {system_prompt}\n{prompt}\nA:"
                response = llm(template)[0]
            # parse response
            # parsed_response = parse_response(response)

            if args.verbose:
                print("--------------------------------------------------")
                print(f"Instruction: {system_prompt}")
                print(f"Story: {prompt}")
                print(f"A: {response}")
                # print(f"Parsed A: {parsed_response}")

            # append to list
            predicted_answers.append(response)
            # parsed_answers.append(parsed_response)

    df_out = pd.DataFrame({
        "affect": affect_lists,
        "goal": goal_lists,
        "affect_valence": affect_valences,
        "utterance": utterances_lists,
        "state": states_list,
        "item": item_lists,
        "predicted_answer": predicted_answers,
        # "parsed_answer": parsed_answers
    })
    # write to file
    if not os.path.exists(os.path.join(args.output_dir, filename)):
        os.makedirs(os.path.join(args.output_dir, filename))

    prefix = f"{args.model.replace('/','_')}_speaker_test_{args.temperature}_{args.num}_{args.offset}_iter{iter}"
    df_out.to_csv(os.path.join(args.output_dir, filename, f"{prefix}_predicted_answers_free_production.csv"), index=False)


# TODO: approach 2: Llama log probability results
