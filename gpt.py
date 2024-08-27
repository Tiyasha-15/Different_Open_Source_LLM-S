from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_article_gpt(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=500, do_sample=True, temperature=0.7)
    article = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return article

prompt = "The impact of AI on modern society"
print(generate_article_gpt(prompt))
