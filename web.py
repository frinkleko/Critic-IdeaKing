from flask import Flask, request, jsonify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 初始化 Flask 应用
app = Flask(__name__)

# 初始化 tokenizer 和 LLM
tokenizer = AutoTokenizer.from_pretrained('output')
sampling_params = SamplingParams(temperature=0.8, top_p=0.8, repetition_penalty=1.05, max_tokens=4096)
llm = LLM(model='output')
# Prepare your prompts
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

system = "You are an AI journal conference reviewer from openreview. Review the following paper's abstract and provide feedback."
instruction = "You are an AI journal conference reviewer from openreview. You need to read the abstract of a " \
              "paper and then review the paper as a reviewer " \
              "to give a rating on the IDEA or other metrics. You need to grade like a real reviewer as follows MarkDown " \
              "format:\n" \
              + markdown_content + \
              "Review the following paper's abstract and provide feedback.\n" \
              "[Abstract]:\n"

def extract_rating(text):
    import re
    # Define the regular expression to find the rating value
    rating_pattern = re.compile(r'## Rating\n(\d+):')
    
    # Search for the pattern in the input text
    match = rating_pattern.search(text)
    
    # If a match is found, return the rating value as an integer
    if match:
        return float(match.group(1))
    else:
        # If no match is found, return None or an appropriate default value
        print('Reward Missing!')
        print("-"*30+'text'+"-"*30)
        print(text)
        return 0


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    messages = data.get('messages', [])
    
    # 处理输入消息
    message_content = "\n".join(messages)

    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": instruction}, {"role": "user", "content": message_content}],
        tokenize=False,
        add_generation_prompt=True
    )

    # 生成输出
    outputs = llm.generate([text], sampling_params)
    generated_texts = []
    for output in outputs:
        generated_texts.append(output.outputs[0].text)
    
    rewards = []
    for text in generated_texts:
        rewards.append(extract_rating(text)/10)
        
    print(rewards)

    return jsonify({"rewards": rewards})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)