from transformers import AutoTokenizer, AutoModelForCausalLM

# Load LLaMA model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

def generate_article_llama(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=500, do_sample=True, temperature=0.7)
    article = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return article

prompt = "The impact of AI on modern society"
print(generate_article_llama(prompt))
