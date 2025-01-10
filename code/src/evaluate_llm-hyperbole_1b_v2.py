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
import pandas as pd

def add_emoji(story, emoji='☺️'):
    story_parsed = story.replace("said", "replied")
    # add emoji at end of the speaker utterance
    story_split = story_parsed.split('."\nQ:') 
    story_wEmoji = story_split[0] + emoji + '."\nQ:' + story_split[1]
    return story_wEmoji

def construct_story(r, prompt_type="prob"):
    """
    Helper for constructing a story
    from u, s, context.
    """
    # TODO: deal with dynamic k in {1,2,3} sampling?
    story = r['context'] + '. ' + r['question'] + '"' + r["sentence"] + r["utterance"] + '." '
    if prompt_type == "prob":
        story_prompt = story + r["probability_prompt"]
    elif prompt_type == "likert":
        story_prompt = story + r["likert_prompt"]
    return story_prompt
    
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
            parsed_response = int(input("Enter response(0-1):"))
            return parsed_response
    elif "answer:" in raw_response.lower():
        response = raw_response.split("answer:")[1].lower().strip()
        return response
    else:
        print(f"Response: {raw_response}")
        parsed_response = 10000
        return parsed_response


parser = argparse.ArgumentParser()

# model args
parser.add_argument('--model', type=str, default='gpt-4o-mini', help='model name')
parser.add_argument('--temperature', type=float, default=0.0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=10, help='max tokens')
parser.add_argument('--prompt', type=str, default="0shot", help='prompt')
parser.add_argument('--num_completions', type=int, default=20, help='number of completions')

# eval args
parser.add_argument('--num', '-n', type=int, default=50, help='number of evaluations')
parser.add_argument('--offset', '-o', type=int, default=0, help='offset')
parser.add_argument('--verbose', action='store_true', help='verbose')

# data args (I need to set up the input and output directory of my data)
parser.add_argument('--data_dir', type=str, default='../../data/', help='data directory')
parser.add_argument('--output_dir', type=str, default='../../data/results_pt/', help='output directory')
parser.add_argument('--datafile', type=str, default='experiment_1b_raw', help='output directory')
parser.add_argument('--promptfile', type=str, default='evaluation_0shot_1b_v3.txt', help='output directory')


# parse args
args = parser.parse_args()


# no condiction is needed or I could change it to ["electric kettle", "watch", "laptop"] and construct the code in a different way

# read data (data should just be a list)

datafile = args.datafile
data = pd.read_csv(os.path.join(args.data_dir, f"{datafile}.csv"))


# get prompt

PROMPT_DIR = "../prompt_instructions/"
if args.prompt == "0shot":
    with open(os.path.join(PROMPT_DIR, args.promptfile), 'r') as f:
        prompt = f.read().strip()
# elif args.prompt == "0shot_cot":
#     with open(os.path.join(PROMPT_DIR, "evaluation_0shot_cot_1b.txt"), 'r') as f:
#         prompt = f.read().strip()
else:
    raise ValueError(f"Prompt {args.prompt} not found.")
    
    
# initialize LLM (Don't need to change)
if args.model in ["gpt-4-0613", "gpt-3.5-turbo", "gpt-4o-mini"]:
    # llm = ChatOpenAI(model_name=args.model,
    #                 temperature=args.temperature,
    #                 max_tokens = args.max_tokens)
    llm = init_model(model_name=args.model,
                    temperature=args.temperature,
                    max_tokens = args.max_tokens) # num_completions=args.num_completions
elif args.model in ["claude-2"]:
    llm = ChatAnthropic(model_name=args.model,
                    temperature=args.temperature,
                    max_tokens = args.max_tokens,
                    num_completions=args.num_completions)
elif args.model in ["llama-2-7b-chat"]:
    llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Llama-2-7b-chat-hf",
        task="text-generation",
        model_kwargs={"temperature": args.temperature, "max_length": args.max_tokens, "num_return_sequences": args.num_completions},
    )
else:
    raise ValueError(f"Model {args.model} not found.")
    
# evaluate (I need to replace the code below to let it compatible with hyperbole dataset)


# no condiction is needed or I could change it to ["electric kettle", "watch", "laptop"]

predicted_answers = []
graded_answers = []
stories = []
# I should pay attention to the args.num as it controls the number of cases will be evaluated
for i in tqdm(range(args.offset, len(data))):
    story = data.iloc[i]
    query = construct_story(story)
    stories.append(story)
    if args.model in ["gpt-4-0613", "gpt-3.5-turbo", "claude-2", "gpt-4o-mini"]:
        messages = [SystemMessage(content=prompt), HumanMessage(content=query)]
        chat_result = llm.generate([messages], stop=["Q:"]).generations[0]
        response = [chat_result[i].text for i in range(len(chat_result))]
    elif args.model in ["llama-2-7b-chat"]:
        template = f"Instructions: {prompt}\n{query}\nA:"
        response = llm(template)[0]
    # parse response
    parsed_response = ", ".join([str(parse_response(r)) for r in response])

    if args.verbose:
        print("--------------------------------------------------")
        print(f"Prompt: {prompt}")
        print(f"Story: {query}")
        print(f"A: {response}")
        print(f"Parsed A: {parsed_response}")

    # append to list
    predicted_answers.append(", ".join(response))
    graded_answers.append(parsed_response)

# write to file
if not os.path.exists(os.path.join(args.output_dir, datafile)):
    os.makedirs(os.path.join(args.output_dir, datafile))

# format output as csv file
results_df = pd.DataFrame({
    "story": stories,
    "predicted_answer": predicted_answers,
    "parsed_answer": graded_answers
})
prefix = f"{args.model.replace('/','_')}_{args.promptfile.replace('.txt', '')}_{args.temperature}_{args.num}_{args.offset}"

# write
results_df.to_csv(os.path.join(args.output_dir, datafile, f"{prefix}_predicted_answers.csv"), index=False)