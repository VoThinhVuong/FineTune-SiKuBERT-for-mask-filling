from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load the tokenizer and model
model_folder = "extended_sikubert_model"
tokenizer = BertTokenizer.from_pretrained(model_folder)
model = BertForMaskedLM.from_pretrained(model_folder)

# Prepare input text
input_text = "固茹員外[MASK]王"

inputs = tokenizer(input_text, return_tensors="pt")

# Perform prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the masked token predictions

mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
mask_token_logits = logits[0, mask_token_index, :]
top_k = 5  # Top 5 predictions
top_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()

# Decode predictions
print("Top predictions for the masked token:")
for token in top_tokens:
    print("Ground: 户 ", "Prediction: ", tokenizer.decode([token]))
