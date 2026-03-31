import pandas as pd
import numpy as np
import torch
import random
from transformers import set_seed, AutoProcessor, AutoModelForImageTextToText, LlavaForConditionalGeneration
import base64
from collections import defaultdict
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    arg_model = parser.add_argument("--model", required=True, 
                       help="which LLM?")
    arg_seed = parser.add_argument("--seed", required=True, 
                       help="How many seeds?")
    return parser.parse_args()

def get_data(filename, batch):
    data = pd.read_csv(filename)
    data = data.drop_duplicates("name")
    return data[data["model_batch"] == batch]

def encode_image(image_id):
    image_path = f"./Images/{image_id}.png"
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_model(model_id):
    if "llava" in model_id.lower():
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            dtype=torch.float16, 
            device_map="auto",
            low_cpu_mem_usage=True, 
        ).to(0)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
    
    return processor, model

def get_scores(input_ids, logits):

    shifted_logits = logits[:, :-1, :] # Ignore the last token's logits
    shifted_labels = input_ids[:, 1:] # Skip the first token in the labels

    # Compute log probabilities
    log_probs = torch.log_softmax(shifted_logits, dim=-1)

    # Gather the log probabilities for the correct tokens
    token_logprobs = log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)

    # 1. Get sequence probabilities
    seq_logprobs = token_logprobs.sum(dim=-1).item()

    # 2. Get sequence probabilities normalized by len(sequence)
    seq_logprobs_norm = token_logprobs.sum() / shifted_labels.size(1) #denominator: len of shifted input_ids
    
    # 3. Get perplexity score for each sequence
    mean_perplexity_score = torch.exp(-seq_logprobs_norm).item()

    # 4. Get PenLP
    pen_lp = seq_logprobs / ( ( (5+shifted_labels.size(1)) / (5 + 1) ) ** 0.8 )

    # get 2 out of torch tensor
    seq_logprobs_norm = seq_logprobs_norm.item()

    return seq_logprobs, seq_logprobs_norm, mean_perplexity_score, pen_lp


def run_model(model_id, batch, seed):

    model_id_short = model_id.split("/")[-1]
    dat = get_data(filename, batch)
    set_seed(seed)
    processor, model = load_model(model_id)

    outcome = defaultdict(dict)

    # -------------- Go over data -----------------
    for _, row in dat.iterrows():
        name = row["name"]
        base64_image_r = encode_image(row["im_r"])
        base64_image_i = encode_image(row["im_i"])

        messages_n = [
            {"role": "user", 
             "content": [{"type": "text", "text": f"""{row["content"]}"""}] 
             } 
            ]

        messages_r = [
            {"role": "user", 
             "content": [{"type": "image", "image": f"data:image/png;base64,{base64_image_r}"},
                         {"type": "text", "text": f"""{row["content"]}"""}]},
            ]

        messages_i = [
            {"role": "user", 
             "content": [{"type": "image", "image": f"data:image/png;base64,{base64_image_i}"},
                         {"type": "text", "text": f"""{row["content"]}"""}]},
            ]

        # -------------- Pass data into model ------------------------------------

        # ---------------------- Null condition -----------------------------------
        inputs_n = processor.apply_chat_template( 
                messages_n,
                tokenize=True,
                add_generation_prompt=True,
                return_dict = True,
                return_tensors="pt"
            ).to(model.device)
        # returns ['input_ids', 'attention_mask']

        with torch.no_grad():
            outputs_n = model(**inputs_n)

        logits_n = outputs_n.logits

        # -------------------- Relevant condition ----------------------------------
        inputs_r = processor.apply_chat_template(
                messages_r,
                tokenize=True,
                add_generation_prompt=True,
                return_dict = True,
                return_tensors="pt"
            ).to(model.device)
        # returns ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']

        with torch.no_grad():
            outputs_r = model(**inputs_r)

        logits_r = outputs_r.logits

        #----------------------- Irrelevant condition ------------------------------
        inputs_i = processor.apply_chat_template(
                messages_i,
                tokenize=True,
                add_generation_prompt=True,
                return_dict = True,
                return_tensors="pt"
            ).to(model.device)
        # returns ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']

        with torch.no_grad():
            outputs_i = model(**inputs_i)

        logits_i = outputs_i.logits


        # -------------- Get logits for a sequence only (exclude image part) -------------------

        sent_index_r = torch.isin(inputs_r.input_ids, inputs_n.input_ids).nonzero(as_tuple=True)
        sent_index_i = torch.isin(inputs_i.input_ids, inputs_n.input_ids).nonzero(as_tuple=True)

        # additionally remove the first three indices 0, 1, 2 -- common across all data points
        logits_n = logits_n
        logits_r = torch.index_select(logits_r,dim=1,
                                      index=torch.tensor(sent_index_r[-1][3:], dtype=torch.long))
        logits_i = torch.index_select(logits_i,dim=1,
                                      index=torch.tensor(sent_index_i[-1][3:], dtype=torch.long))

        input_ids_n = inputs_n.input_ids
        input_ids_r = torch.index_select(inputs_r.input_ids, dim=1, 
                                         index=torch.tensor(sent_index_r[-1][3:], dtype=torch.long))
        input_ids_i = torch.index_select(inputs_i.input_ids, dim=1,
                                         index=torch.tensor(sent_index_i[-1][3:], dtype=torch.long))

        # -------------- Get logprobs, norm logprobs, perplexity for N, R, I conditions ---------------------

        seq_logprobs_n, seq_logprobs_norm_n, perp_n, penlp_n = get_scores(input_ids_n, logits_n)
        seq_logprobs_r, seq_logprobs_norm_r, perp_r, penlp_r = get_scores(input_ids_r, logits_r)
        seq_logprobs_i, seq_logprobs_norm_i, perp_i, penlp_i = get_scores(input_ids_i, logits_i)

        outcome["n_logprob"][name] = seq_logprobs_n
        outcome["r_logprob"][name] = seq_logprobs_r
        outcome["i_logprob"][name] = seq_logprobs_i

        outcome["n_logprob_norm"][name] = seq_logprobs_norm_n
        outcome["r_logprob_norm"][name] = seq_logprobs_norm_r
        outcome["i_logprob_norm"][name] = seq_logprobs_norm_i

        outcome["n_perp"][name] = perp_n
        outcome["r_perp"][name] = perp_r
        outcome["i_perp"][name] = perp_i

        outcome["n_penlp"][name] = penlp_n
        outcome["r_penlp"][name] = penlp_r
        outcome["i_penlp"][name] = penlp_i


    # ------------------------ Save data ------------------------------
    outcome = pd.DataFrame(outcome)
    outcome["model"] = model_id_short
    outcome["batch"] = batch
    outcome["seed"] = seed

    outcome.to_csv(output_dir + f"/scores_{model_id_short}_{batch}_{seed}.csv")

# ------------------------------------------------------
def main():

    args = parse_args()
    input_model = args.model
    if "qwen" in input_model.lower() and "3b" in input_model.lower():
        model_id =  "Qwen/Qwen2.5-VL-3B-Instruct"
    elif "qwen" in input_model.lower() and "7b" in input_model.lower():
        model_id =  "Qwen/Qwen2.5-VL-7B-Instruct"
    elif "llava" in input_model.lower():
        model_id = "llava-hf/llava-1.5-7b-hf"
    elif "internvl" in input_model.lower() and "1b" in input_model.lower():
        model_id = "OpenGVLab/InternVL3-1B-hf"
    elif "internvl" in input_model.lower() and "8b" in input_model.lower():
        model_id = "OpenGVLab/InternVL3-8B-hf"
    else:
        raise argparse.ArgumentError(arg_model, "Model not supported.")

    num_seed = int(args.seed)

    for batch in batches:
        print(batch)
        seeds = random.sample(range(100, 500), num_seed) 
        for seed in seeds:
            print(seed)
            run_model(model_id, batch, seed)


my_path = os.path.dirname(os.path.realpath(__file__))
print(my_path)
filename = my_path + "/sentences.csv"
batches = ["batch1", "batch2", "batch3", "batch4", "batch5"]

output_dir = os.path.join(my_path, 'ModelLogits')
print(output_dir)

if __name__ == "__main__":
    main()

print("DONE!")



