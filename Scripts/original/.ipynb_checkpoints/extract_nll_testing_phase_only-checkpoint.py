import torch
import pandas as pd
import numpy as np

# --- Configuration ---
DOMAIN              = "Food" # Or set dynamically
INPUT_PATH          = f'../Data/Model Outputs/log_likelihood_{DOMAIN}_ALL_marcelbinz-Llama-3.1-Centaur-70B-adapter.pth'
OUTPUT_PATH         = f'../Data/Model Outputs/log_likelihood_{DOMAIN}_TESTING_ALIGNED.pth'
NARRATIVE_DATA_PATH = '../Data/Preprocessed Data/narrative_data.csv'

# 1. Load Data
nlls_list = torch.load(INPUT_PATH)
df        = pd.read_csv(NARRATIVE_DATA_PATH)

cleaned_nlls        = []
EXPECTED_TEST_ITEMS = 68

# Filter participants per domain
df_sel = df[df['domain'] == DOMAIN]

print(f"In NLL: {len(nlls_list)} participants...")
print(f"In df: {df_sel.shape[0]} participants...")

for i, participant_nlls in enumerate(nlls_list):
    # Get metadata for this participant
    # Note: Assuming nlls_list order matches the narrative_data.csv row order
    row     = df_sel.iloc[i]
    n_train = int(row['n_training'])
    
    # Parse ID_items: convert "50, 9, ..." into a list of integers
    all_ids  = [int(x.strip()) for x in str(row['ID_items']).split(',')]
    test_ids = all_ids[-EXPECTED_TEST_ITEMS:]
    
    # a) Check and Extract Testing Phase Only
    total_expected = n_train + EXPECTED_TEST_ITEMS
    actual_len     = len(participant_nlls)
    
    if actual_len != total_expected:
        print(f"Warning: Participant {i} (ID: {row['ID']}) length mismatch! "
              f"Expected {total_expected}, found {actual_len}.")
    
    # Slice the last 68 items
    test_nlls = participant_nlls[-EXPECTED_TEST_ITEMS:]
    
    # b) Reorder according to ID_item
    # We create a mapping of {ItemID: NLL_Value}
    # Then we sort that mapping by ItemID (1 to 80, though we only have 68 test items)
    id_nll_pairs = list(zip(test_ids, test_nlls.tolist()))
    
    # Sort pairs by the ID (the first element of the tuple)
    # This ensures Item ID 1 is first, Item ID 2 is second, etc.
    id_nll_pairs.sort(key=lambda x: x[0])
    
    # Extract just the NLLs back into a tensor
    sorted_nlls = torch.tensor([pair[1] for pair in id_nll_pairs])
    
    cleaned_nlls.append(sorted_nlls)

# Save the final list of tensors
torch.save(cleaned_nlls, OUTPUT_PATH)

print(f"Success! Saved aligned NLLs to {OUTPUT_PATH}")
print(f"Each participant now has {cleaned_nlls[0].shape[0]} NLLs ordered by Item ID.")