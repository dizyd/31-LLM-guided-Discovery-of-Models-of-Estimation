import re
import gc
import torch
from tqdm import tqdm


# ── CHECK 1: Does the model output a valid number between << and >>? ──────────
# Feeds a prompt ending with " <<" and checks that the model completes it
# with a number (e.g. "75>>") rather than text.

def check_prediction_format(model, tokenizer, dataset, device, n_samples=5):
    for i in range(n_samples):
        text = dataset[i]['text']

        # Cut the text right after the last " <<" so the model must fill in the blank
        cut = text.rfind(' <<')
        prompt = text[:cut + len(' <<')]

        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False)

        generated = tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predicted = generated.split('>>')[0].strip()  # grab what came before ">>"

        try:
            float(predicted)
            print(f"  Sample {i}: ✓  model predicted '{predicted}'")
        except ValueError:
            print(f"  Sample {i}: ✗  model generated '{generated}' — not a number!")


# ── CHECK 2: Do response numbers tokenize as a single token? ──────────────────
# Responses with multiple tokens (e.g. "1500" -> ["15", "00"]) are fine for
# NLL computation (the sum of token NLLs still equals -log p(response)),
# but this check makes the splitting visible so you know it's happening.

def check_tokenization_consistency(tokenizer, dataset):
    multi_token = {}  # response string -> token breakdown

    for row in dataset:
        for resp in re.findall(r'<<(.*?)>>', row['text']):
            ids = tokenizer(' ' + resp, add_special_tokens=False).input_ids
            if len(ids) > 1:
                multi_token[resp] = [tokenizer.decode([t]) for t in ids]

    if not multi_token:
        print("✓ All responses are single-token.")
    else:
        print(f"⚠ {len(multi_token)} unique responses split into multiple tokens:")
        for resp, tokens in list(multi_token.items())[:10]:
            print(f"  '{resp}' -> {tokens}")


# ── CORRECTED NLL LOOP ────────────────────────────────────────────────────────
# Same as the notebook loop, but returns one NLL value *per response* (i.e. one
# per << ... >> pair) rather than one per token. This makes NLL values directly
# comparable across responses regardless of how many tokens each number uses.

def compute_nll_per_response(model, tokenizer, dataloader, device):
    l_id = tokenizer(' <<', add_special_tokens=False).input_ids
    r_id = tokenizer('>>',  add_special_tokens=False).input_ids
    all_nlls = []  # one tensor per participant, each entry = NLL of one response

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Participants"):
            outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            targets = batch['labels'][0, 1:].cpu()

            # Per-token NLL (same as notebook), zeroed out where labels == -100
            token_nll = torch.nn.functional.cross_entropy(
                outputs.logits[0, :-1].cpu(), targets, reduction='none'
            )
            token_nll[targets == -100] = 0.0

            # Sum token NLLs within each << ... >> span -> one value per response
            seq = batch['input_ids'][0].tolist()
            response_nlls = []
            j = 0
            while j < len(seq):
                if seq[j:j + len(l_id)] == l_id:
                    start = j + len(l_id)
                    for k in range(start, len(seq)):
                        if seq[k:k + len(r_id)] == r_id:
                            # token_nll[i] predicts seq[i+1], so shift by -1
                            response_nlls.append(token_nll[start - 1:k - 1].sum().item())
                            j = k + len(r_id)
                            break
                    else:
                        j += 1
                else:
                    j += 1

            all_nlls.append(torch.tensor(response_nlls))

            del outputs, targets, token_nll
            torch.cuda.empty_cache()
            gc.collect()

    return all_nlls


# ── USAGE (paste into notebook after loading model/tokenizer/dataset) ─────────
#
# check_prediction_format(model, tokenizer, dataset, device, n_samples=5)
# check_tokenization_consistency(tokenizer, dataset)
#
# nlls = compute_nll_per_response(model, tokenizer, dataloader, device)
# # nlls[i] is a 1-D tensor with one NLL per response for participant i
# torch.save(nlls, f'Results/nll_{DOMAIN}_{MODEL.replace("/", "-")}.pth')
