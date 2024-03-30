# Simple demonstration on how an LLM generates text--one token at a time, using the previous tokens to predict the
# following ones.

""" Step 1. Load a tokenizer and a model """
# First, loading a tokenizer and a model from HuggingFace's transformers library. (A tokenizer is a function that
# splits a string into a list of numbers that the model can understand). Tokens in this case are neither just
# letters nor just words (subwords).


from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pretrained model and a tokenizer using HuggingFace. Using the GPT-2 model for this example.
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# We create a partial sentence and tokenize it.
text = "Generative AI can be both"
inputs = tokenizer(text, return_tensors="pt")

# Show the tokens as numbers, i.e. "input_ids"
inputs["input_ids"]
# %%
# Show how the sentence is tokenized, Using pandas for better visualization.
import pandas as pd


def show_tokenization(inputs):
    return pd.DataFrame(
        [(id, tokenizer.decode(id)) for id in inputs["input_ids"][0]],
        columns=["id", "token"],
    )


show_tokenization(inputs)

# %%
""" Step 2. Calculate the probability of the next token using pytorch """
# Calculate the probabilities for the next token for all possible choices and show top 5 choices and the corresponding
# words or subwords for these tokens.

import torch

with torch.no_grad():
    logits = model(**inputs).logits[:, -1, :]
    probabilities = torch.nn.functional.softmax(logits[0], dim=-1)


def show_next_token_choices(probabilities, top_n=5):
    return pd.DataFrame(
        [
            (id, tokenizer.decode(id), p.item())
            for id, p in enumerate(probabilities)
            if p.item()
        ],
        columns=["id", "token", "p"],
    ).sort_values("p", ascending=False)[:top_n]


show_next_token_choices(probabilities)
# %%
# Obtain the token id for the most probable next token
next_token_id = torch.argmax(probabilities).item()

print(f"Next token id: {next_token_id}")
print(f"Next token: {tokenizer.decode(next_token_id)}")
# %%
# Append the most likely token to the text.
text = text + tokenizer.decode(next_token_id)
text
# %%
""" Step 3. Generate some more tokens """
# The following cell will take `text`, show the most probable tokens to follow, and append the most likely token to
# text. Run the cell over and over to see it in action

from IPython.display import Markdown, display


# Show the text
print(text)

# Convert to tokens
inputs = tokenizer(text, return_tensors="pt")

# Calculate the probabilities for the next token and show the top 5 choices
with torch.no_grad():
    logits = model(**inputs).logits[:, -1, :]
    probabilities = torch.nn.functional.softmax(logits[0], dim=-1)

print("Next token probabilities")
display(show_next_token_choices(probabilities))

# Choose the most likely token id and add it to the text
next_token_id = torch.argmax(probabilities).item()
text = text + tokenizer.decode(next_token_id)

# %%
""" Step 4. Use the generate method """

inputs = tokenizer(text, return_tensors="pt")

# Use the `generate` method to generate lots of text
output = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)

# Show the generated text
display(tokenizer.decode(output[0]))
