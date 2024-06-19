import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set your Hugging Face token
os.environ['HF_TOKEN'] = "hf_huggingface token"

def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
    )
    return tokenizer, model

def generate_response(system_message, user_message, tokenizer, model):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

    response = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(response, skip_special_tokens=True)

def summarize_text(text):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer, model = load_model_and_tokenizer(model_id)
    
    system_message = (
        "You are a chatbot that performs summarization. Summarize the key content using "
        "one of the following words: Where does it hurt, No fever, High fever, "
        "Name and birthdate, Administer anesthesia, "
        "After meals, Before bed, Before meals, Check blood pressure, Check temperature, "
        "Evening, Fine, Get injection, Headache, Lunchtime, Morning, Need surgery, Normal, "
        "Sign surgical consent. Choose from these words to express it simply. If multiple are included, let me know."
    )
    
    return generate_response(system_message, text, tokenizer, model)

if __name__ == "__main__":
    # Example usage
    original_text = 'Please sit and take blood pressure after take temperature'
    summary_text = summarize_text(original_text)
    print(summary_text)
