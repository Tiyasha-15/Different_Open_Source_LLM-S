from transformers import AutoTokenizer, AutoModelForCausalLM

# Load BLOOM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

def generate_article_bloom(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=500, do_sample=True, temperature=0.7)
    article = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return article

prompt = "The impact of AI on modern society"
print(generate_article_bloom(prompt))
