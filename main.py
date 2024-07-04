import re
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from flask import Flask, request, jsonify


markdown_content = """
    # Review

    ## Summary
    
    ## Soundness

    ## Presentation

    ## Contribution

    ## Strengths
    
    ## Weaknesses

    ## Rating
    
    ## Questions
    
    ## Suggestions, Ideas, and Comments
    
    ## Limitations

    ## Ethics Review

    ## Confidence
    
    """


def clean_sentence(sentence):
    # Remove tab characters
    sentence = sentence.replace('\t', ' ')

    # Allow Markdown syntax characters and remove any other character that is not a letter, digit, punctuation,
    # or whitespace
    sentence = re.sub(r'[^\w\s\d.,!?;:()#\[\]_\-*/]', '', sentence)

    # Replace multiple spaces with a single space
    sentence = re.sub(r'\s+', ' ', sentence)

    # Strip leading and trailing whitespace
    sentence = sentence.strip()

    return sentence


def create_data():
    instruction = "You are a highly creative researcher in the field of Deep Learning and Machine Learning, and you need to come upwith ideas" \
    "for a publishable paper that are as innovative as possible, and that can beconsidered innovative enough to" \
    "achieve a high score in openreview under the criteria of journal conference reviewers, such as ICML, ICLR and NeurIPS's reviewers. The form of the idea you" \
    "generate is an abstract of a paper. For example:\n[Abstract]: Language models are increasingly being deployed" \
    "for general problem solving across a wide range of tasks, but are still confined to token-level, left-to-right" \
    "decision-making processes during inference. This means they can fall short in tasks that require exploration," \
    "strategic lookahead, or where initial decisions play a pivotal role. To surmount these challenges, we introduce" \
    "a new framework for language model inference, Tree of Thoughts (ToT), which generalizes over the popular Chain of" \
    "Thought approach to prompting language models, and enables exploration over coherent units of text (thoughts) that" \
    "serve as intermediate steps toward problem solving. ToT allows LMs to perform deliberate decision making by considering" \
    "multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking" \
    "ahead or backtracking when necessary to make global choices.\nOur experiments show that ToT significantly enhances" \
    "language models’ problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24," \
    "Creative Writing, and Mini Crosswords. For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only" \
    "solved 4\\% of tasks, our method achieved a success rate of 74\\%. \nNow generate your idea next!\n[Abstract]:"
              
    # 示例数据
    example = {
            "instruction": instruction,
            "input": "",
            "output": ""
        }

    num_items = 100
    data = []
    for i in range(num_items):
        data.append(example)

    # 保存到本地JSON文件
    output_file = 'data/Ideas.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"JSON file with {num_items} items has been saved to {output_file}")
    
    
def critic():
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('output')

    # Pass the default decoding hyperparameters of Qwen1.5-7B-Chat
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=0.9, top_p=0.8, repetition_penalty=1.05, max_tokens=4096)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model='output')
    
    instruction = "You are an AI journal conference reviewer from openreview. You need to read the abstract of a " \
                "paper and then review the paper as a reviewer " \
                "to give a rating on the IDEA or other metrics. You need to grade like a real reviewer as follows MarkDown " \
                "format:\n" \
                + markdown_content + \
                "Review the following paper's abstract and provide feedback.\n" \
                "[Abstract]:\n"
    while True:
        prompt = input('input: ')
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # generate outputs
        outputs = llm.generate([text], sampling_params)

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(generated_text)
            
            
def king():
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('king')

    # Pass the default decoding hyperparameters of Qwen1.5-7B-Chat
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model='king')

    instruction = "You are a highly creative researcher in the field of Deep Learning and Machine Learning, and you need to come upwith ideas" \
    "for a publishable paper that are as innovative as possible, and that can beconsidered innovative enough to" \
    "achieve a high score in openreview under the criteria of journal conference reviewers, such as ICML, ICLR and NeurIPS's reviewers. The form of the idea you" \
    "generate is an abstract of a paper. For example:\n[Abstract]: Language models are increasingly being deployed" \
    "for general problem solving across a wide range of tasks, but are still confined to token-level, left-to-right" \
    "decision-making processes during inference. This means they can fall short in tasks that require exploration," \
    "strategic lookahead, or where initial decisions play a pivotal role. To surmount these challenges, we introduce" \
    "a new framework for language model inference, Tree of Thoughts (ToT), which generalizes over the popular Chain of" \
    "Thought approach to prompting language models, and enables exploration over coherent units of text (thoughts) that" \
    "serve as intermediate steps toward problem solving. ToT allows LMs to perform deliberate decision making by considering" \
    "multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking" \
    "ahead or backtracking when necessary to make global choices.\nOur experiments show that ToT significantly enhances" \
    "language models’ problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24," \
    "Creative Writing, and Mini Crosswords. For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only" \
    "solved 4\\% of tasks, our method achieved a success rate of 74\\%. \nNow generate your idea next!\n[Abstract]:"
                
    while True:
        prompt = input('input: ')
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": ""}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # generate outputs
        outputs = llm.generate([text], sampling_params)

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(generated_text)
            
            
def Eval(mode='iclr'):
    name = input('conference: ')
    year = input('year: ')
    
    path = 'conferences/'+name+year+'.json'
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('output')

    # Pass the default decoding hyperparameters of Qwen1.5-7B-Chat
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.8, repetition_penalty=1.05, max_tokens=1024)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model='output')
    
    instruction = "You are an AI journal conference reviewer from openreview. You need to read the abstract of a " \
              "paper and then review the paper as a reviewer " \
              "to give a rating on the IDEA or other metrics. You need to grade like a real reviewer as follows MarkDown " \
              "format:\n" \
              + markdown_content + \
              "Review the following paper's abstract and provide feedback.\n" \
              "[Abstract]:\n"
    
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    ground_truth = []    
    predict = []    
    for index in range(0, len(data), 100):
        item = data[index]
        
        messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": clean_sentence(item['input'])}
            ]
        
        text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # generate outputs
        outputs = llm.generate([text], sampling_params)
        
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            break
        
        print(generated_text)
        
        iclr_pattern = re.compile(r'## Recommendation\n(\d+):')
        nips_pattern = re.compile(r'## Rating\n(\d+):')
        
        # Search for the pattern in the input text
        iclr_match = iclr_pattern.search(generated_text)
        nips_match = nips_pattern.search(generated_text)
        
        # If a match is found, return the rating value as an integer
        if iclr_match:
            score1 =  float(iclr_match.group(1))
        elif nips_match:
            score1 =  float(nips_match.group(1))
        else:
            # If no match is found, return None or an appropriate default value
            print('Predict Missing!')
            continue
            
        # Search for the pattern in the input text
        iclr_match = iclr_pattern.search(item['output'])
        nips_match = nips_pattern.search(item['output'])
        
        # If a match is found, return the rating value as an integer
        if iclr_match:
            score2 =  float(iclr_match.group(1))
        elif nips_match:
            score2 =  float(nips_match.group(1))
        else:
            # If no match is found, return None
            print('Groundtruth Missing!')
            continue
            
        predict.append(score1)
        ground_truth.append(score2)
        print(index, '/', len(data), 'finished!')
        
    print('Size equal:', len(predict) == len(ground_truth))
    
    # Ensure inputs are numpy arrays
    y_true = np.array(ground_truth)
    y_pred = np.array(predict)
    
    # Calculate MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Calculate MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate SMAPE
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    print(f'Result of {path}')
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"SMAPE: {smape}")  
      
    # Create KDE plots for true and predicted values
    sns.kdeplot(y_true, label='True Values', shade=True)
    sns.kdeplot(y_pred, label='Predicted Values', shade=True)
    
    # Add titles and labels
    plt.title(path)
    plt.xlabel('Rating')
    plt.ylabel('Density')
    plt.legend()
    
    # Save the plot to a file
    plt.savefig('pics/'+name.lower()+'_'+year.lower()+'.jpg')
    plt.close()  # Close the figure to free memory
    
    print('Saved to', 'pics/'+name.lower()+'_'+year.lower()+'.jpg')
    
                 
if __name__ == '__main__':
    critic()
    # king()
    # create_data()
    Eval(mode="nips")
    # plot()
    