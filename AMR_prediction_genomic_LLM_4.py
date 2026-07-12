# Generated from: AMR_prediction_genomic_LLM_4.ipynb
# Converted at: 2026-07-12T18:49:12.865Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Predicting Antimicrobial Resistance (AMR) from Bacterial Gene Sequences
# ### Using a pretrained genomic language model (Nucleotide Transformer)
# 
# **What this notebook does:** given a raw DNA sequence of a bacterial gene, predict which
# antibiotic drug class it most likely confers resistance to — using embeddings from a
# pretrained genomic language model instead of hand-built features.
# 
# **Pipeline:**
# 1. Download & extract the CARD (Comprehensive Antibiotic Resistance Database) reference data
# 2. Parse sequences & build a labelled dataset: gene sequence → drug class
# 3. Baseline: k-mer frequency + Logistic Regression (sanity check)
# 4. Load a pretrained genomic LLM (InstaDeep's Nucleotide Transformer)
# 5. Extract embeddings for every sequence
# 6. Train a lightweight classifier head on those embeddings
# 7. Evaluate (accuracy, per-class recall, confusion matrix) and compare to the baseline
# 8. Interpretability: occlusion-based importance — which part of the sequence drives the prediction
# 9. Predict on your own pasted-in DNA sequence
# 10. Look up documented resistance genes by organism name
# 11. Write-up notes, citations, and next steps
# 
# **Before you start (Colab):** `Runtime > Change runtime type > T4 GPU` (free tier is fine).
# Everything below is written to run end-to-end on the free tier, using a lightweight 100M-parameter
# model rather than the larger 500M/2.5B versions.


# ## Step 0 — Install dependencies
# **Important — pinned `transformers` version, on purpose.** The Nucleotide Transformer models are
# loaded via `trust_remote_code=True`, which pulls in a custom `modeling_esm.py` file from the model's
# own Hugging Face repo. That custom file still imports a helper function
# (`find_pruneable_heads_and_indices`) from `transformers.pytorch_utils`, but newer `transformers`
# releases (5.x) removed/relocated it — and not every Nucleotide Transformer model size has had its
# custom code updated to match. Installing 'latest' `transformers` will therefore break model loading
# in Step 4 with an `ImportError`. Pinning to a known-good pre-5.0 version avoids that entirely.
# 
# **If you already ran this notebook once with a newer `transformers` installed:** after this cell
# finishes, go to `Runtime > Restart session` and then run all cells again from the top — pip
# installing a different version has no effect on an already-running Python process.


!pip install -q 'transformers==4.53.0' accelerate biopython scikit-learn matplotlib seaborn pandas tqdm

import torch
print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))

import transformers
print('transformers version:', transformers.__version__)

# ## Step 1 — Download and extract the CARD database
# [CARD](https://card.mcmaster.ca/) (Alcock et al., *Nucleic Acids Research*, 2023) is the standard
# curated database of antimicrobial resistance genes — free for academic/research use (see the
# [CARD download page](https://card.mcmaster.ca/download) for the full usage terms; commercial
# redistribution needs separate permission from McMaster University, academic use does not).
# 
# We use the auto-download endpoint, which always points at the latest release.


import os, tarfile, urllib.request

os.makedirs('card_data', exist_ok=True)
card_url = 'https://card.mcmaster.ca/latest/data'
archive_path = 'card_data/card_data.tar.bz2'

if not os.path.exists(archive_path):
    urllib.request.urlretrieve(card_url, archive_path)

with tarfile.open(archive_path, 'r:bz2') as tar:
    tar.extractall('card_data')

print(sorted(os.listdir('card_data'))[:20])

# ## Step 2 — Parse the sequences and build a labelled dataset
# We use `nucleotide_fasta_protein_homolog_model.fasta` — the reference nucleotide sequences for
# CARD's "protein homolog" model type (i.e. full-length known resistance genes, not point-mutation
# variants). Each FASTA header looks like:
# 
# `>gb|AJ920369|+|23-860|ARO:3001071|SHV-12 [Escherichia coli]`
# 
# We extract the ARO accession from the header and join it against `aro_index.tsv`, which has the
# curated **Drug Class** label for every ARO accession — this becomes our prediction target. We also
# keep the organism name in `[brackets]` at the end of the header — that's the species this specific
# reference sequence was originally isolated from and deposited under in GenBank. We'll use it later
# (Step 10) to look genes up by organism name.


import re
import pandas as pd
from Bio import SeqIO

fasta_path = 'card_data/nucleotide_fasta_protein_homolog_model.fasta'
aro_index_path = 'card_data/aro_index.tsv'

records = []
aro_pattern = re.compile(r'ARO:(\d+)')
organism_pattern = re.compile(r'\[(.*?)\]\s*$')  # organism name is in [brackets] at the end of the header
for rec in SeqIO.parse(fasta_path, 'fasta'):
    match = aro_pattern.search(rec.description)
    if match:
        organism_match = organism_pattern.search(rec.description)
        records.append({
            'aro_accession': 'ARO:' + match.group(1),
            'sequence': str(rec.seq).upper(),
            'length': len(rec.seq),
            'organism': organism_match.group(1) if organism_match else 'Unknown',
        })

seq_df = pd.DataFrame(records)
aro_index = pd.read_csv(aro_index_path, sep='\t')
aro_index = aro_index.rename(columns={'ARO Accession': 'aro_accession', 'Drug Class': 'drug_class'})

data = seq_df.merge(aro_index[['aro_accession', 'drug_class']], on='aro_accession', how='left')
data = data.dropna(subset=['drug_class'])

# A gene can be annotated with multiple drug classes separated by '; ' —
# we simplify to a single-label problem by taking the first listed class.
# (Note this in your write-up: it's a deliberate simplification, not multi-label modelling.)
data['primary_drug_class'] = data['drug_class'].str.split(';').str[0].str.strip()

print(f'Total labelled sequences: {len(data)}')
data['primary_drug_class'].value_counts().head(15)

# Keep it tractable: classify only the most frequent drug classes.
TOP_N_CLASSES = 8
top_classes = data['primary_drug_class'].value_counts().head(TOP_N_CLASSES).index.tolist()
data_top = data[data['primary_drug_class'].isin(top_classes)].reset_index(drop=True)

# Also drop very short/very long outlier sequences to keep tokenization sane
data_top = data_top[(data_top['length'] >= 200) & (data_top['length'] <= 3000)].reset_index(drop=True)

print(f'Sequences after filtering: {len(data_top)}')
print(f'Classes: {top_classes}')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data_top['label'] = le.fit_transform(data_top['primary_drug_class'])

train_df, test_df = train_test_split(
    data_top, test_size=0.2, stratify=data_top['label'], random_state=42
)
print(f'Train: {len(train_df)}  Test: {len(test_df)}')

# ## Step 3 — Baseline: k-mer frequency + Logistic Regression
# Before reaching for the language model, build a simple baseline. If the LLM-embedding model
# can't beat this, that's an important (and honest) finding to report — not something to hide.


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def kmer_tokenize(seq, k=4):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

vectorizer = CountVectorizer(analyzer=lambda s: kmer_tokenize(s, k=4))
X_train_kmer = vectorizer.fit_transform(train_df['sequence'])
X_test_kmer = vectorizer.transform(test_df['sequence'])

baseline_clf = LogisticRegression(max_iter=2000)
baseline_clf.fit(X_train_kmer, train_df['label'])
baseline_preds = baseline_clf.predict(X_test_kmer)

print('--- Baseline (k-mer + Logistic Regression) ---')
print(classification_report(test_df['label'], baseline_preds, target_names=le.classes_, zero_division=0))

# ## Step 4 — Load the pretrained genomic language model
# We use [`InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-100m-multi-species)
# — a 100M-parameter transformer pretrained on 850 genomes across many species. It's the smallest
# model in the Nucleotide Transformer v2 family, chosen specifically so this runs on a free Colab GPU.
# If you have access to more compute later, swapping in `nucleotide-transformer-v2-500m-multi-species`
# is a one-line change and a natural 'ablation' to report.
# 
# (Reminder: this only works with the pinned `transformers==4.53.0` from Step 0 — if you see
# `ImportError: cannot import name 'find_pruneable_heads_and_indices'`, Step 0 wasn't run with the
# pin, or the runtime wasn't restarted after installing it.)


from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL_NAME = 'InstaDeepAI/nucleotide-transformer-v2-100m-multi-species'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
llm_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
llm_model.eval()

# Quick sanity check on a dummy sequence
test_tokens = tokenizer(['ATTCCGATTCCGATTCCG'], return_tensors='pt', padding=True)['input_ids'].to(device)
with torch.no_grad():
    out = llm_model(test_tokens, output_hidden_states=True)
print('Hidden states shape:', out['hidden_states'][-1].shape)

# ## Step 5 — Extract embeddings for every sequence
# We mean-pool the last hidden layer over real (non-padding) tokens to get one fixed-size vector
# per gene sequence. `MAX_TOKENS` caps sequence length for speed — raise it if your sequences are
# longer and you have the compute budget; the model's own hard cap is 1000 tokens.


import numpy as np
from tqdm.auto import tqdm

MAX_TOKENS = 400   # ~2,400 bp at 6 nt/token — covers most single resistance genes
BATCH_SIZE = 8

def embed_sequences(sequences, max_tokens=MAX_TOKENS, batch_size=BATCH_SIZE):
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch = sequences[i:i + batch_size]
        tokens = tokenizer.batch_encode_plus(
            batch, return_tensors='pt', padding='max_length',
            truncation=True, max_length=max_tokens
        )['input_ids'].to(device)
        attention_mask = (tokens != tokenizer.pad_token_id)
        with torch.no_grad():
            outputs = llm_model(
                tokens, attention_mask=attention_mask,
                encoder_attention_mask=attention_mask, output_hidden_states=True
            )
        hidden = outputs['hidden_states'][-1]
        mask = attention_mask.unsqueeze(-1)
        mean_embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(mean_embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)

X_train_emb = embed_sequences(train_df['sequence'].tolist())
X_test_emb = embed_sequences(test_df['sequence'].tolist())

np.save('X_train_emb.npy', X_train_emb)
np.save('X_test_emb.npy', X_test_emb)
print('Train embeddings shape:', X_train_emb.shape)

# ## Step 6 — Train a classifier head on the LLM embeddings
# The pretrained model's weights stay frozen — we're only training a small classifier on top of
# its embeddings ('probing'). This is realistic on free-tier compute and is standard practice
# when working with foundation models on a student budget.


from sklearn.neural_network import MLPClassifier

llm_clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42)
llm_clf.fit(X_train_emb, train_df['label'])
llm_preds = llm_clf.predict(X_test_emb)

print('--- Nucleotide Transformer embeddings + MLP classifier ---')
print(classification_report(test_df['label'], llm_preds, target_names=le.classes_, zero_division=0))

# ## Step 7 — Compare baseline vs. LLM-embedding model
# Accuracy alone is misleading for imbalanced classes — look at **per-class recall** (missing a
# resistant gene class is the costly error) alongside the confusion matrix.


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

print(f"Baseline accuracy:    {accuracy_score(test_df['label'], baseline_preds):.3f}")
print(f"LLM-embedding accuracy: {accuracy_score(test_df['label'], llm_preds):.3f}")
print(f"Baseline macro-F1:    {f1_score(test_df['label'], baseline_preds, average='macro'):.3f}")
print(f"LLM-embedding macro-F1: {f1_score(test_df['label'], llm_preds, average='macro'):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, preds, title in zip(
    axes, [baseline_preds, llm_preds], ['Baseline (k-mer + LogReg)', 'Nucleotide Transformer + MLP']
):
    cm = confusion_matrix(test_df['label'], preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_,
                yticklabels=le.classes_, ax=ax, cbar=False)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()

# ## Step 8 — Interpretability: which part of the sequence drives the prediction?
# We use **occlusion**: slide a window across the sequence, mask it out (replace with `N`s), and
# measure how much the model's confidence in the correct class drops. A big drop means that region
# mattered a lot — in a real write-up, check whether high-importance regions line up with known
# catalytic or resistance-conferring motifs for that gene family.
# 
# This is model-agnostic (works regardless of whether the underlying transformer exposes attention
# weights cleanly), which makes it a safer default for a first project.


def occlusion_importance(sequence, true_label_idx, window=30, stride=15):
    positions, drops = [], []
    baseline_emb = embed_sequences([sequence], batch_size=1)
    baseline_prob = llm_clf.predict_proba(baseline_emb)[0][true_label_idx]
    for start in range(0, len(sequence) - window, stride):
        occluded = sequence[:start] + ('N' * window) + sequence[start + window:]
        occ_emb = embed_sequences([occluded], batch_size=1)
        occ_prob = llm_clf.predict_proba(occ_emb)[0][true_label_idx]
        positions.append(start)
        drops.append(baseline_prob - occ_prob)
    return positions, drops

# Run on one example test sequence
example_row = test_df.iloc[0]
positions, drops = occlusion_importance(example_row['sequence'], example_row['label'])

plt.figure(figsize=(12, 4))
plt.plot(positions, drops, marker='o', markersize=3)
plt.axhline(0, color='gray', linewidth=0.8)
plt.xlabel('Sequence position (bp)')
plt.ylabel('Drop in predicted probability when occluded')
plt.title(f"Occlusion importance — {example_row['primary_drug_class']} gene")
plt.tight_layout()
plt.show()

# ## Step 9 — Input your own DNA sequence
# Everything so far ran on sequences pulled automatically from CARD. **This is where you paste in a
# sequence of your own** and get a prediction from the trained model.
# 
# Paste a raw nucleotide sequence as a plain string (just the letters A, T, G, C — no FASTA header,
# no line breaks). If you're starting from a `.fasta` file, open it in a text editor, copy everything
# after the `>header` line, and join the wrapped lines into one continuous string first.


def predict_drug_class(raw_sequence, verbose=True):
    # Clean up: uppercase, strip whitespace/newlines in case it was pasted from a wrapped FASTA file
    sequence = ''.join(raw_sequence.split()).upper()
    valid_bases = set('ACGTN')
    if not set(sequence).issubset(valid_bases):
        bad_chars = set(sequence) - valid_bases
        raise ValueError(f'Sequence contains non-DNA characters: {bad_chars}')

    embedding = embed_sequences([sequence], batch_size=1)
    probabilities = llm_clf.predict_proba(embedding)[0]

    ranked = sorted(zip(le.classes_, probabilities), key=lambda x: x[1], reverse=True)
    if verbose:
        print(f'Sequence length: {len(sequence)} bp\n')
        print('Predicted drug class ranking:')
        for drug_class, prob in ranked:
            print(f'  {drug_class:25s} {prob:.1%}')
    return ranked[0][0]

# ### Try it on a real sequence first
# Before pasting in anything of your own, sanity-check the function on a sequence you already have
# — one pulled straight from your own held-out test set. Its true label is known, so you can check
# whether the prediction matches.


# Grab one real example from the test set you already built in Step 2
example = test_df.sample(1, random_state=7).iloc[0]
print(f"True drug class: {example['primary_drug_class']}\n")

predicted = predict_drug_class(example['sequence'])
print(f"\nMatch: {predicted == example['primary_drug_class']}")

# ### Now try your own sequence
# Paste a raw nucleotide sequence as a plain string (just the letters A, T, G, C — no FASTA header,
# no line breaks). If you're starting from a `.fasta` file, open it in a text editor, copy everything
# after the `>header` line, and join the wrapped lines into one continuous string first.


# --- Paste your own sequence below and run this cell ---
my_sequence = 'PASTE_YOUR_DNA_SEQUENCE_HERE'

# predict_drug_class(my_sequence)  # <- uncomment once you've pasted a real sequence above

# ## Step 10 — Look up resistance by organism name
# Instead of pasting a DNA sequence, search by **bacterium name** and see which resistance genes
# CARD has on file for that species, and what drug classes they're linked to.
# 
# **Important — what this does and doesn't mean.** Each CARD reference sequence is tagged with the
# organism it was originally isolated from when someone deposited it in GenBank. So this tells you
# *'these are resistance genes that have been documented in strains of this species'* — it is
# **not** a guarantee that every individual bacterium of that species carries them, and it isn't a
# live susceptibility result for any specific culture you're holding. Think of it as a documented
# history for the species, not a diagnosis for one sample.


def lookup_organism_resistance(organism_query, cross_check_with_model=True, max_check=5):
    matches = data[data['organism'].str.contains(organism_query, case=False, na=False)]

    if matches.empty:
        print(f"No CARD reference genes tagged with an organism containing '{organism_query}'.")
        print("Try a shorter/partial name — e.g. 'coli' instead of 'Escherichia coli'.")
        return

    print(f"Found {len(matches)} CARD reference gene(s) tagged with organism containing '{organism_query}'\n")
    print('Documented drug classes among these genes (per CARD curation):')
    print(matches['primary_drug_class'].value_counts().to_string())

    if not cross_check_with_model:
        return

    # The trained classifier only knows the drug classes it was trained on (top_classes from Step 2).
    # Only genes in those classes can be cross-checked against the model's own prediction.
    checkable = matches[matches['primary_drug_class'].isin(top_classes)]
    if checkable.empty:
        print("\n(None of these genes fall within the drug classes the model was trained on,"
              " so no model cross-check is available — see Step 2's TOP_N_CLASSES.)")
        return

    sample_n = min(max_check, len(checkable))
    print(f"\nCross-checking {sample_n} gene(s) against the trained model's own prediction:\n")
    for _, row in checkable.sample(sample_n, random_state=1).iterrows():
        predicted = predict_drug_class(row['sequence'], verbose=False)
        match_mark = 'match' if predicted == row['primary_drug_class'] else 'differs'
        print(f"  {row['aro_accession']:15s} CARD label: {row['primary_drug_class']:22s} "
              f"Model says: {predicted:22s} ({match_mark})")


# Example — try any bacterium name you like
lookup_organism_resistance('Escherichia coli')

# Try other species by changing the name — e.g. `lookup_organism_resistance('Staphylococcus aureus')`,
# `lookup_organism_resistance('Pseudomonas aeruginosa')`, or a genus-only partial match like
# `lookup_organism_resistance('Klebsiella')`. If a search comes back empty, the organism may just not
# appear in the `protein_homolog` model subset of CARD we loaded — try a shorter/partial name first.


# ## Write-up notes, citations, and next steps
# 
# **Suggested framing for your thesis/report:**
# *“In-silico prediction of antimicrobial resistance drug class from gene sequence using a
# pretrained genomic language model, benchmarked against a k-mer baseline.”*
# 
# **Things worth reporting honestly:**
# - How much (if at all) the LLM-embedding model beats the k-mer baseline — report both, even if
#   the gap is small. That comparison is exactly what makes this credible rather than a black box.
# - The single-label simplification (multi-drug-class genes reduced to their first listed class)
#   — state it explicitly as a limitation.
# - Class imbalance — note which drug classes had the fewest sequences and interpret their recall
#   cautiously.
# 
# **Natural next step:** Step 9 (paste-your-own-sequence) and Step 10 (search-by-organism) are both
# there so you can explore beyond CARD's built-in examples once the pipeline is working end-to-end.
# 
# **Key citations:**
# - Alcock et al. (2023). *CARD 2023: expanded curation, support for machine learning, and
#   resistome prediction at the Comprehensive Antibiotic Resistance Database.* Nucleic Acids Research.
# - Dalla-Torre et al. (2023). *The Nucleotide Transformer: Building and Evaluating Robust
#   Foundation Models for Human Genomics.* bioRxiv.