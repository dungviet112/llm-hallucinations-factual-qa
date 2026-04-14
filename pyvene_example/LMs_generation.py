import torch
import pyvene as pv

# built-in helper to get tinystore
_, tokenizer, tinystory = pv.create_gpt_neo()
emb_happy = tinystory.transformer.wte(
    torch.tensor(14628)) 

pv_tinystory = pv.IntervenableModel([{
    "layer": l,
    "component": "mlp_output",
    "intervention_type": pv.AdditionIntervention
    } for l in range(tinystory.config.num_layers)],
    model=tinystory
)
# prompt and generate
prompt = tokenizer(
    "Once upon a time there was", return_tensors="pt")
unintervened_story, intervened_story = pv_tinystory.generate(
    prompt, source_representations=emb_happy*0.3, max_length=256
)

print(tokenizer.decode(
    intervened_story[0], 
    skip_special_tokens=True
))