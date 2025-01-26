from openai import OpenAI
import os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
class MyGPT:
    def __init__(self, model_name='gpt-4o-mini', temp=0.6, top_p=0.9):
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=openai_api_key)
        self.temp = temp
        self.top_p = top_p
        self.model_name = model_name
    def query(self, message):
        messages = [
            {'role': 'user', 'content': message},
        ]
        print(messages)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temp,
            max_tokens=300,
            top_p=self.top_p,
            messages=messages
        )
        return completion.choices[0].message.content


def generate_response_llama(prompts, model_name='meta-llama/Meta-Llama-3-70B-Instruct', temp=0.6, top_p=0.9, save_size=2000, out_dir='temp'):
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Initialize the pipeline once
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # Calculate the number of batches
    total_prompts = len(prompts)
    total_batches = (total_prompts + save_size - 1) // save_size
    print("\ntotal_batches = ", total_batches)
    # Process each batch
    for batch_num in tqdm(range(total_batches), desc="batches"):
        start_idx = batch_num * save_size
        end_idx = min((batch_num + 1) * save_size, total_prompts)
        batch_prompts = prompts[start_idx:end_idx]

        # Check if this batch file already exists
        batch_file = os.path.join(out_dir, f'llama3_responses_{batch_num}.csv')
        if os.path.exists(batch_file):
            print(f"Batch {batch_num} already exists, skipping...")
            continue

        response_dict = {'prompt': [], 'response': []}

        # Generate responses for this batch
        for prompt in tqdm(batch_prompts, desc=f"Processing batch {batch_num}", unit="prompt"):
            messages = [{"role": "user", "content": prompt}]

            terminators = [ pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

            outputs = pipeline( messages, max_new_tokens=300, eos_token_id=terminators, do_sample=True, temperature=temp, top_p=top_p)
            # The pipeline returns a list of responses, 
            # the last character is indexed [-1] for the generated text.
            answer = outputs[0]["generated_text"][-1]
            # print("--------------------Answer: ", answer)
            response_dict['prompt'].append(prompt)
            response_dict['response'].append(answer)

        # Save the batch results to a CSV
        df = pd.DataFrame(response_dict)
        df.to_csv(batch_file, index=False)
        print(f"Saved batch {batch_num} to {batch_file}")

def generate_response_mixtral7x8(prompts, model_name='mistralai/Mixtral-8x7B-Instruct-v0.1', temp=0.6, top_p=0.9, save_size=2000, batch_size=64, max_new_tokens=300, out_dir='temp'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad token if not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    response_dict = {'prompt': [], 'response': []}
    num = 0  # Counter for file naming

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size)):
        # for i in tqdm(range(0, 16, batch_size)):
            batch_prompts = [f"[INST]{prompt}[/INST]" for prompt in prompts[i : i + batch_size]]

            # Tokenize
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True
            ).to(model.device)
            
            # Generate responses
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=top_p
            )

            # Decode and append responses
            for prompt, output in zip(batch_prompts, outputs):
                response_text = tokenizer.decode(output, skip_special_tokens=True).strip()
                response_dict['prompt'].append(prompt)
                response_dict['response'].append(response_text)

            # Save periodically
            if len(response_dict['prompt']) >= save_size:
                df = pd.DataFrame(response_dict)
                df.to_csv(f'{out_dir}/mixtral7x8_responses_{num}.csv', index=False)
                response_dict = {'prompt': [], 'response': []}  # Reset for next batch
                num += 1

            del inputs, outputs
            torch.cuda.empty_cache()


        # Save any remaining data
        if response_dict['prompt']:
            df = pd.DataFrame(response_dict)
            df.to_csv(f'{out_dir}/mixtral7x8_responses_{num}.csv', index=False)



def generate_response_alpaca7B(prompts, model_name='PKU-Alignment/alpaca-7b-reproduced', temp=0.6, top_p=0.9, save_size=2000, out_dir='temp'):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    response_dict = {'prompt': [], 'response': []}
    i = 0
    num = 0
    for prompt in tqdm(prompts):
        if i < 4000:
            i += 1
            continue
        prompt = f'BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:'
        input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
        output_ids = model.generate(input_ids, max_new_tokens=300, do_sample=True, temperature=temp, top_p=top_p)[0]

        response_dict['response'].append(tokenizer.decode(output_ids, skip_special_tokens=True).split('ASSISTANT:')[1])
        response_dict['prompt'].append(prompt)
        print(f'Number of responses: {len(response_dict["response"])}')
        if len(response_dict['prompt'])%save_size == 0:
            df = pd.DataFrame(response_dict)
            df.to_csv(f'{out_dir}/alpaca7B_responses_{num}.csv', index=False)
            num += 1
            response_dict = {'prompt': [], 'response': []}
    if response_dict:
        df = pd.DataFrame(response_dict)
        num = len(response_dict['prompt'])//save_size
        df.to_csv(f'{out_dir}/alpaca7B_responses_{num}.csv', index=False)



def generate_response_llama2(prompts, model_name='meta-llama/Llama-2-7b-chat-hf', temp=0.6, top_p=0.9, save_size=2000, out_dir='temp'):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    response_dict = {'prompt': [], 'response': []}
    num = 0
    for prompt in tqdm(prompts):
        messages = [
            {"role": "user", "content": prompt},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=300,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
        )
        response_dict['response'].append(outputs[0]["generated_text"][-1]['content'])
        response_dict['prompt'].append(prompt)
        if len(response_dict['prompt'])%save_size == 0:
            df = pd.DataFrame(response_dict)
            df.to_csv(f'{out_dir}/llama2_responses_{num}.csv', index=False)
            num += 1
            response_dict = {'prompt': [], 'response': []}
    if response_dict:
        df = pd.DataFrame(response_dict)
        num = len(response_dict['prompt'])//save_size
        df.to_csv(f'{out_dir}/llama2_responses_{num}.csv', index=False)



def generate_response_mistral7B(prompts, model_name='mistralai/Mistral-7B-Instruct-v0.3', temp=0.6, top_p=0.9, save_size=2000, out_dir='temp'):
    chatbot = transformers.pipeline("text-generation", 
                                    model=model_name,
                                    model_kwargs={"torch_dtype": torch.bfloat16},
                                    device_map="auto")
    response_dict = {'prompt': [], 'response': []}
    num = 0
    for prompt in tqdm(prompts):
        messages = [
            {"role": "user", "content": prompt},
        ]
        outputs = chatbot(messages, max_new_tokens=300, do_sample=True, temperature=temp, top_p=top_p)
        response_dict['response'].append(outputs[0]["generated_text"][-1]['content'])
        response_dict['prompt'].append(prompt)
        if len(response_dict['prompt'])%save_size == 0:
            df = pd.DataFrame(response_dict)
            df.to_csv(f'{out_dir}/mistral7B_responses_{num}.csv', index=False)
            response_dict = {'prompt': [], 'response': []}
            num += 1
    if response_dict:
        df = pd.DataFrame(response_dict)
        df.to_csv(f'{out_dir}/mistral7B_responses_{num}.csv', index=False)

def generate_response_gpt4omini(prompts, temp=0.6, top_p=0.9, save_size=2000, out_dir='temp'):
    gpt = MyGPT(temp=temp, top_p=top_p)
    response_dict = {'prompt': [], 'response': []}
    num = 0
    for prompt in tqdm(prompts):
        response_dict['response'].append(gpt.query(prompt))
        response_dict['prompt'].append(prompt)
        if len(response_dict['prompt'])%save_size == 0:
            df = pd.DataFrame(response_dict)
            df.to_csv(f'{out_dir}/gpt4omini_responses_{num}.csv', index=False)
            response_dict = {'prompt': [], 'response': []}
            num += 1
    if response_dict:
        df = pd.DataFrame(response_dict)
        df.to_csv(f'{out_dir}/gpt4omini_responses_{num}.csv', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, default='gpt-4o-mini', help="name of model to be run")
    parser.add_argument('-t', '--temp', type=float, default=0.6, help="temperature")
    parser.add_argument('-p', '--top_p', type=float, default=0.9, help="top_p")
    parser.add_argument('-s', '--save_size', type=int, default=2000, help="number of prompts to save")
    parser.add_argument('-o', '--output_path', type=str, default='.', help="path to save output")
    args = parser.parse_args()
    out_dir = args.output_path + '/' + args.model_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    if args.model_name == 'gpt-4o-mini':
        print(generate_response_gpt4omini(temp=args.temp, top_p=args.top_p, save_size=args.save_size, out_dir=out_dir))
    elif args.model_name == 'mistral7B':
        print(generate_response_mistral7B(temp=args.temp, top_p=args.top_p, save_size=args.save_size, out_dir=out_dir))
    elif args.model_name == 'llama2':
        print(generate_response_llama2(temp=args.temp, top_p=args.top_p, save_size=args.save_size, out_dir=out_dir))
    elif args.model_name == 'alpaca7B':
        print(generate_response_alpaca7B(temp=args.temp, top_p=args.top_p, save_size=args.save_size, out_dir=out_dir))
    elif args.model_name == 'mixtral7x8':
        print(generate_response_mixtral7x8(temp=args.temp, top_p=args.top_p, save_size=args.save_size, out_dir=out_dir))
    elif args.model_name == 'llama3':
        print(generate_response_llama(temp=args.temp, top_p=args.top_p, save_size=args.save_size, out_dir=out_dir))
    else:
        print(f'Invalid model name: {args.model_name}')