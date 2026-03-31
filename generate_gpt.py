import pandas as pd
import random
import base64
from glob import glob
import argparse
from openai import OpenAI

def parse_args():
    parser = argparse.ArgumentParser()
    arg_model = parser.add_argument("--model", required=True, 
                       help="which LLM? (gpt-4o or gpt-4o-mini).")
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

def run_model_no_image(dat, model, seed):
    responses = {}
    for _, row in dat.iterrows():
        name = row["name"]

        completion = client.chat.completions.create(
                  model=model,
                    seed=seed,
                  messages=[
                          {
                            "role": "user",
                            "content": f""" On a scale of 1 (not at all) - 2 (not so much) - 3 (somewhat) - 4 (completely), how natural is this sentence? Only say a number and nothing else: "{row["content"]}" """
                          } 
                  ]
      
    )
    
        responses[name] = completion.choices[0].message.content
        
    responses = pd.DataFrame(responses.items(), columns=['name', 'natural'])
    responses["model"] = model
    responses["seed"] = seed
    responses["batch"] = batch
    responses["condition"] = condition

    responses.to_csv(output_dir + f"/preds_{model}_{batch}{condition}_{seed}.csv", index=False)


def run_model_with_image(dat, model, seed, condition):
    responses1 = {}
    responses2 = {}
    for _, row in dat.iterrows():
        name = row["name"]
        if condition == "r":
            base64_image = encode_image(row["im_r"])
            a, b, c = row["choices_r"].split(",")
        elif condition == "i":
            base64_image = encode_image(row["im_i"])
            a, b, c = row["choices_i"].split(",")      
        
        # image task                
        context = [
            {
                "role": "user",
                "content": [
                    { 
                     "type": "text", 
                     "text": f"""which of the following is at the most front of this image: 
                      {a}, {b}, {c}? Only say {a}, {b}, or {c} and nothing else.""" 
                    },
             
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ],
            } 
        ]
        
        response = client.chat.completions.create(
            model= model,
            seed=seed,
            messages=context
        )
    
        responses1[name] = response.choices[0].message.content
        
        # sentence acceptability task
        context+=  [            
                {
                    "role": "user",
                    "content": [
                        { "type": "text", 
                          "text":  f""" On a scale of 1 (not at all) - 2 (not so much) - 3 (somewhat) - 4 (completely), how natural is this sentence? Only say a number and nothing else:
                                    "{row["content"]}" """ },
                    ],
            } 
        ]

        response2 = client.chat.completions.create(
            model= model,
            seed=seed,
            messages=context
        )

        responses2[name] = response2.choices[0].message.content

    responses = pd.DataFrame(responses1.items(), columns=['name', 'choice'])
    responses["natural"] = responses["name"].map(responses2)
    responses["model"] = model
    responses["seed"] = seed
    responses["batch"] = batch
    responses["condition"] = condition

    responses.to_csv(output_dir + f"/preds_{model}_{batch}{condition}_{seed}.csv", index=False)


def main():
    dat = get_data(filename, batch)
    args = parse_args()

    # condition
    condition = args.condition
    if condition.lower() not in ["r", "i", "n"]:
        raise argparse.ArgumentError(arg_condition, 
        "Unsupported condition. Choose from 'r', 'i', or 'n'.")
    print(condition)

    # seed
    num_seed = int(args.seed)

    # model
    model_id = args.model
    
    for batch in batches:
        seeds = random.sample(range(100, 500), num_seed) 
        for seed in seeds:
            if condition == "r":
                run_model_with_image(dat, model=model_id, seed=seed, condition=condition)
            elif condition == "i": 
                run_model_with_image(dat, model=model_id, seed=seed, condition=condition)
            elif condition == "n":
                run_model_no_image(dat, model=model_id, seed=seed)
    
#------------------------------------------------------------------
with open("api_key.txt", "r") as file:
    my_key = file.read()
client = OpenAI(api_key=my_key)

batches = ["batch1", "batch2", "batch3", "batch4", "batch5"]

my_path = os.path.dirname(os.path.realpath(__file__))
print("My path:", my_path)
filename = my_path + "/sentences.csv"
image_dir = my_path + "/Images/"

output_dir = os.path.join(my_path, 'ModelPredictions')
print("Output directory:", output_dir)

if __name__ == "__main__":
    main()

print("DONE!")



