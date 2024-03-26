import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

def load_model(model_id):
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

# See https://qdrant.tech/articles/sparse-vectors/ for code and
# https://huggingface.co/naver/splade-v3 for model.
def expand(model, tokenizer, text):
    tokens = tokenizer(text, return_tensors="pt")
    output = model(**tokens)
    logits = output.logits
    mask = tokens.attention_mask
    matrix = torch.log(1 + torch.relu(logits)) * mask.unsqueeze(-1)
    vector = torch.max(matrix, dim=1).values.squeeze(0)
    ids = vector.nonzero().squeeze(-1)
    weights = vector[ids]
    indices = torch.argsort(weights)
    sorted_ids = ids[indices].tolist()
    sorted_tokens = tokenizer.convert_ids_to_tokens(sorted_ids)
    sorted_weights = weights[indices].tolist()
    return sorted_tokens, sorted_weights

# See https://plotly.com/javascript/ for API and
# https://d3js.org/d3-format for formatting language.
def plot(text, tokens, weights, n):
    tokens = tokens[-n:]
    weights = weights[-n:]
    html = f"""\
<!DOCTYPE html>
<html>
<head>
<title>SPLADE demo</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
</head>
<body>
<h1>{text}</h1>
<div id="plot"></div>
<script>
var data = [{{
  y: {tokens},
  x: {weights},
  type: 'bar',
  orientation: 'h',
  hovertemplate: '%{{y}} %{{x}}<extra></extra>',
  xhoverformat: '.2f'
}}];
var layout = {{
  width: 800,
  height: 600
}};
Plotly.newPlot('plot', data, layout);
</script>
</body>
</html>"""
    return html

model_id = "naver/splade-v3"
text = "Do I have to register with your office to be protected?"

model, tokenizer = load_model(model_id)
tokens, weights = expand(model, tokenizer, text)
html = plot(text, tokens, weights, 20)
print(html)
