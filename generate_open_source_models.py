import pandas as pd
import numpy as np
import torch
import random
from transformers import set_seed, AutoProcessor, AutoModelForImageTextToText, LlavaForConditionalGeneration
import base64
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    arg_model = parser.add_argument("--model", required=True, 
                       help="which LLM? (qwen s, qwen b, internvl s, internvl b, llava).")
    arg_seed = parser.add_argument("--seed", required=True, 
                       help="How many seeds? Give an integer value.")
    arg_condition = parser.add_argument("--condition", required=True,
                        help="Which condition? (n, r, i)")
    return parser.parse_args()

def get_data(filename, batch):
    data = pd.read_csv(filename)
    data = data.drop_duplicates("name")

    return data[data["model_batch"] == batch]

def encode_image(image_id):
    image_path = image_dir + f"{image_id}.png"
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

def run_model_no_image(model_id, seed, batch, temp):

    model_id_short = model_id.split("/")[-1]
    temp = round(temp, 2)
    dat = get_data(filename, batch)
    set_seed(seed)

    processor, model = load_model(model_id)
    natural = {}

    for index, row in dat.iterrows():
        name = row["name"]
        
        messages = [
            {
        "role": "user",
        "content": [

            {"type": "text", "text": f""" On a scale of 1 (not at all) - 2 (not so much) - 3 (somewhat) - 4 (completely), how natural is this sentence? Only say a number and nothing else: "{row["content"]}" """}
                                      
            ]
        },
    ],

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict = True,
            return_tensors="pt"
        ).to(model.device)

        # Run inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=temp,
                top_p = 1.0,
                top_k = 0,
                do_sample=True
            )
      # Decode output
        response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:],
                                    skip_special_tokens=True).split("\n")[-1] 
        natural[name] = response

    responses = pd.DataFrame(natural.items(), columns=['name', 'natural'])

    responses["model"] = model_id_short
    responses["batch"] = batch
    responses["condition"] = "n"
    responses["seed"] = seed
    responses["temperture"] = temp
    responses.to_csv(output_dir + f"/preds_{model_id_short}_{batch}n_temp{temp}_{seed}.csv", index=False)



def run_model_with_image(model_id, seed, batch, temp, condition):

    model_id_short = model_id.split("/")[-1]
    temp = round(temp, 2)
    dat = get_data(filename, batch)
    set_seed(seed)

    processor, model = load_model(model_id)

    choices = {}
    natural = {}
    for index, row in dat.iterrows():
        name = row["name"]
        if condition == "r":
            base64_image = encode_image(row["im_r"])
            a, b, c = row["choices_r"].split(",")
        elif condition == "i":
            base64_image = encode_image(row["im_i"])
            a, b, c = row["choices_i"].split(",")            

        # image question
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{base64_image}"},
                    {"type": "text", "text": f"""which of the following is at the most front of this image: {a}, {b}, {c}? Only say {a}, {b}, or {c} and nothing else."""}            
                    ]
                    }
        ],

        inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict = True,
                return_tensors="pt"
            ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=temp,
                top_p = 1.0,
                top_k = 0,
                do_sample=True
            )
            
        # Decode output
        choice = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], 
                                  skip_special_tokens=True).split("\n")[-1] 
        choices[name] = choice

        # sentence rating following image question
        messages = [
            
            {"role": "user", 
             "content": [{"type": "image", "image": f"data:image/png;base64,{base64_image}"},
                         {"type": "text", "text": f"""which of the following is at the most front of this image: {a}, {b}, {c}? Only say {a}, {b}, or {c} and nothing else."""}]},

            {"role": "assistant", "content": [{"type": "text", "text": choice}]}, 
            {"role": "user", "content": [{"type": "text", 
                                           "text": f""" On a scale of 1 (not at all) - 2 (not so much) - 3 (somewhat) - 4 (completely), how natural is this sentence? Only say a number and nothing else: "{row["content"]}" """}]}
            
            ]

        inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict = True,
                return_tensors="pt"
            ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=temp,
                top_p = 1.0,
                top_k = 0,
                do_sample=True
            )

        # Decode output
        response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], 
                                    skip_special_tokens=True).split("\n")[-1] 
        natural[name] = response

    natural = pd.DataFrame(natural.items(), columns=['name', 'natural'])
    choices = pd.DataFrame(choices.items(), columns=['name', 'choice'])
    responses = natural.merge(choices, on="name")
    
    responses["model"] = model_id_short
    responses["batch"] = batch
    responses["condition"] = condition
    responses["seed"] = seed
    responses["temperture"] = temp

    responses.to_csv(output_dir + f"/preds_{model_id_short}_{batch}{condition}_temp{temp}_{seed}.csv",
                     index=False)


def main():
    args = parse_args()

    # model
    input_model = args.model
    if "qwen" in input_model.lower() and "s" in input_model.lower():
        model_id =  "Qwen/Qwen2.5-VL-3B-Instruct"
    elif "qwen" in input_model.lower() and "b" in input_model.lower():
        model_id =  "Qwen/Qwen2.5-VL-7B-Instruct"
    elif "llava" in input_model.lower():
        model_id = "llava-hf/llava-1.5-7b-hf"
    elif "internvl" in input_model.lower() and "s" in input_model.lower():
        model_id = "OpenGVLab/InternVL3-1B-hf"
    elif "internvl" in input_model.lower() and "b" in input_model.lower():
        model_id = "OpenGVLab/InternVL3-8B-hf"
    else:
        raise argparse.ArgumentError(arg_model, "Model not supported.")

    print("Model:", model_id)

    # condition
    condition = args.condition
    if condition.lower() not in ["r", "i", "n"]:
        raise argparse.ArgumentError(arg_condition, 
        "Unsupported condition. Choose from 'r', 'i', or 'n'.")
    print(condition)

    # seed
    num_seed = int(args.seed)

    temperatures = np.arange(0.5, 1.001, 0.05) 
    for temp in temperatures:
        for batch in batches:
            print(batch)
            seeds = random.sample(range(100, 500), num_seed) 
            for seed in seeds:
                print(seed)
                if condition.lower() == "r":
                    run_model_with_image(model_id, seed, batch, temp, condition="r")
                elif condition.lower() == "i":
                    run_model_with_image(model_id, seed, batch, temp, condition="i")
                elif condition.lower() == "n":
                    run_model_no_image(model_id, seed, batch, temp)
                elif condition.lower() not in ["r", "i", "n"]:
                    raise argparse.ArgumentError(arg_condition, 
                    "Unsupported condition. Choose from 'r', 'i', or 'n'.")


# ------------------------------------------------------
my_path = os.path.dirname(os.path.realpath(__file__))
print("My path:", my_path)
filename = my_path + "/sentences.csv"
image_dir = my_path + "/Images/"
batches = ["batch1", "batch2", "batch3", "batch4", "batch5"]

output_dir = os.path.join(my_path, 'ModelPredictions')
print("Output directory:", output_dir)

if __name__ == "__main__":
    main()

print("DONE!")


