"""
AMR Drug-Class Predictor -- Streamlit app

Predicts which antibiotic drug class a bacterial gene sequence most likely
confers resistance to, using CARD (Comprehensive Antibiotic Resistance
Database) reference data and embeddings from a pretrained genomic language
model (Nucleotide Transformer).

Deploy on Streamlit Community Cloud:
  1. Put this file and requirements.txt in your GitHub repo
  2. streamlit.io/cloud -> New app -> point at this file
  3. First load will be slow (several minutes) -- it downloads CARD and
     trains a small classifier on top of the language model's embeddings,
     once per running app instance (cached after that via st.cache_resource).

NOTE ON HOSTING RESOURCES: Streamlit Community Cloud's free tier has limited
CPU/RAM and no GPU. If the first load times out or runs out of memory, the
more robust approach is to train the classifier once separately (e.g. in
Google Colab with a GPU), save it, and have this app only do inference by
loading that saved classifier instead of training on every cold start. Ask
if you want that version built instead.
"""

import functools
import os
import re
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

CARD_URL = "https://card.mcmaster.ca/latest/data"
DATA_DIR = "card_data"
DEFAULT_MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
TOP_N_CLASSES = 8
MAX_TOKENS = 400
BATCH_SIZE = 8

st.set_page_config(page_title="AMR Drug-Class Predictor", page_icon="\U0001F9EC")


# ---------------------------------------------------------------------------
# Data pipeline (same logic as the notebook/script version)
# ---------------------------------------------------------------------------
def download_card(data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    archive_path = os.path.join(data_dir, "card_data.tar.bz2")

    if not os.path.exists(archive_path):
        urllib.request.urlretrieve(CARD_URL, archive_path)

    fasta_check = os.path.join(data_dir, "nucleotide_fasta_protein_homolog_model.fasta")
    if not os.path.exists(fasta_check):
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(data_dir)
    return data_dir


def parse_card_data(data_dir=DATA_DIR):
    from Bio import SeqIO

    fasta_path = os.path.join(data_dir, "nucleotide_fasta_protein_homolog_model.fasta")
    aro_index_path = os.path.join(data_dir, "aro_index.tsv")

    aro_pattern = re.compile(r"ARO:(\d+)")
    organism_pattern = re.compile(r"\[(.*?)\]\s*$")

    records = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        match = aro_pattern.search(rec.description)
        if not match:
            continue
        organism_match = organism_pattern.search(rec.description)
        records.append({
            "aro_accession": "ARO:" + match.group(1),
            "sequence": str(rec.seq).upper(),
            "length": len(rec.seq),
            "organism": organism_match.group(1) if organism_match else "Unknown",
        })

    seq_df = pd.DataFrame(records)
    aro_index = pd.read_csv(aro_index_path, sep="\t")
    aro_index = aro_index.rename(columns={"ARO Accession": "aro_accession", "Drug Class": "drug_class"})

    data = seq_df.merge(aro_index[["aro_accession", "drug_class"]], on="aro_accession", how="left")
    data = data.dropna(subset=["drug_class"])
    data["primary_drug_class"] = data["drug_class"].str.split(";").str[0].str.strip()
    return data


def build_train_test(data, top_n_classes=TOP_N_CLASSES, min_len=200, max_len=3000,
                      test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    top_classes = data["primary_drug_class"].value_counts().head(top_n_classes).index.tolist()
    data_top = data[data["primary_drug_class"].isin(top_classes)].reset_index(drop=True)
    data_top = data_top[(data_top["length"] >= min_len) & (data_top["length"] <= max_len)].reset_index(drop=True)

    label_encoder = LabelEncoder()
    data_top["label"] = label_encoder.fit_transform(data_top["primary_drug_class"])

    train_df, test_df = train_test_split(
        data_top, test_size=test_size, stratify=data_top["label"], random_state=random_state
    )
    return train_df, test_df, top_classes, label_encoder


def load_llm(model_name=DEFAULT_MODEL_NAME):
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    return tokenizer, model, device


def embed_sequences(sequences, tokenizer, model, device, max_tokens=MAX_TOKENS, batch_size=BATCH_SIZE):
    import torch

    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        tokens = tokenizer.batch_encode_plus(
            batch, return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_tokens
        )["input_ids"].to(device)
        attention_mask = (tokens != tokenizer.pad_token_id)
        with torch.no_grad():
            outputs = model(
                tokens, attention_mask=attention_mask,
                encoder_attention_mask=attention_mask, output_hidden_states=True
            )
        hidden = outputs["hidden_states"][-1]
        mask = attention_mask.unsqueeze(-1)
        mean_embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(mean_embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def train_llm_classifier(X_train_emb, y_train):
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42)
    clf.fit(X_train_emb, y_train)
    return clf


def predict_drug_class(raw_sequence, embed_fn, clf, label_encoder):
    """Returns a list of (drug_class, probability) tuples, sorted highest first.
    Raises ValueError if the sequence contains non-DNA characters."""
    sequence = "".join(raw_sequence.split()).upper()
    valid_bases = set("ACGTN")
    if not set(sequence).issubset(valid_bases):
        bad_chars = sorted(set(sequence) - valid_bases)
        raise ValueError(f"Sequence contains non-DNA characters: {', '.join(bad_chars)}")
    if len(sequence) == 0:
        raise ValueError("Sequence is empty.")

    embedding = embed_fn([sequence])
    probabilities = clf.predict_proba(embedding)[0]
    ranked = sorted(zip(label_encoder.classes_, probabilities), key=lambda x: x[1], reverse=True)
    return ranked, len(sequence)


def lookup_organism_resistance(organism_query, data, top_classes, embed_fn, clf, label_encoder, max_check=5):
    """
    NOTE on what this does and doesn't mean: each CARD reference sequence is
    tagged with the organism it was originally isolated from when deposited in
    GenBank. This shows genes CARD has documented for that species -- it is
    NOT a guarantee that every individual bacterium of that species carries
    them, and it isn't a live susceptibility result for any specific culture.

    Returns (matches_df, drug_class_counts, cross_check_df or None).
    """
    matches = data[data["organism"].str.contains(organism_query, case=False, na=False)]
    if matches.empty:
        return matches, None, None

    drug_class_counts = matches["primary_drug_class"].value_counts()

    checkable = matches[matches["primary_drug_class"].isin(top_classes)]
    if checkable.empty:
        return matches, drug_class_counts, None

    sample_n = min(max_check, len(checkable))
    sample = checkable.sample(sample_n, random_state=1)
    rows = []
    for _, row in sample.iterrows():
        ranked, _ = predict_drug_class(row["sequence"], embed_fn, clf, label_encoder)
        predicted = ranked[0][0]
        rows.append({
            "ARO accession": row["aro_accession"],
            "CARD label": row["primary_drug_class"],
            "Model prediction": predicted,
            "Agrees?": "Yes" if predicted == row["primary_drug_class"] else "No",
        })
    cross_check_df = pd.DataFrame(rows)
    return matches, drug_class_counts, cross_check_df


# ---------------------------------------------------------------------------
# One-time setup, cached for the life of the running app instance
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(model_name=DEFAULT_MODEL_NAME, top_n_classes=TOP_N_CLASSES):
    download_card()
    data = parse_card_data()
    train_df, _test_df, top_classes, label_encoder = build_train_test(data, top_n_classes=top_n_classes)

    tokenizer, model, device = load_llm(model_name)
    embed_fn = functools.partial(
        embed_sequences, tokenizer=tokenizer, model=model, device=device,
        max_tokens=MAX_TOKENS, batch_size=BATCH_SIZE,
    )
    X_train_emb = embed_fn(train_df["sequence"].tolist())
    llm_clf = train_llm_classifier(X_train_emb, train_df["label"])

    # Single-sequence embedding calls (organism lookup, manual predictions) use batch_size=1
    embed_fn_single = functools.partial(
        embed_sequences, tokenizer=tokenizer, model=model, device=device,
        max_tokens=MAX_TOKENS, batch_size=1,
    )
    return data, top_classes, label_encoder, llm_clf, embed_fn_single


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("\U0001F9EC AMR Drug-Class Predictor")
st.caption(
    "Predicts the likely antibiotic drug class for a bacterial gene sequence, using CARD "
    "reference data and a pretrained genomic language model (Nucleotide Transformer)."
)

with st.spinner("Loading model and training classifier (first run only -- can take several minutes)..."):
    data, top_classes, label_encoder, llm_clf, embed_fn = load_pipeline()

st.success(f"Ready. Trained on {len(top_classes)} drug classes: {', '.join(top_classes)}")

tab_predict, tab_organism = st.tabs(["Predict from a sequence", "Look up by organism"])

with tab_predict:
    st.subheader("Paste a DNA sequence")
    st.caption("Letters A, T, G, C (and N) only -- no FASTA header, no line breaks needed, whitespace is stripped automatically.")
    seq_input = st.text_area("Sequence", height=150, key="seq_input")

    if st.button("Predict drug class", type="primary"):
        if not seq_input.strip():
            st.warning("Please paste a sequence first.")
        else:
            try:
                ranked, seq_len = predict_drug_class(seq_input, embed_fn, llm_clf, label_encoder)
                st.write(f"Sequence length: **{seq_len} bp**")
                result_df = pd.DataFrame(ranked, columns=["Drug class", "Probability"])
                result_df["Probability"] = result_df["Probability"].apply(lambda p: f"{p:.1%}")
                st.dataframe(result_df, hide_index=True, use_container_width=True)
            except ValueError as e:
                st.error(str(e))

with tab_organism:
    st.subheader("Search by bacterium name")
    st.caption(
        "Shows resistance genes CARD has on file for strains of this species -- a documented "
        "history, not a live test result for any specific bacterium you're holding."
    )
    organism_input = st.text_input("Organism name", placeholder="e.g. Escherichia coli", key="organism_input")

    if st.button("Search", type="primary"):
        if not organism_input.strip():
            st.warning("Please enter an organism name.")
        else:
            matches, drug_class_counts, cross_check_df = lookup_organism_resistance(
                organism_input, data, top_classes, embed_fn, llm_clf, label_encoder
            )
            if matches.empty:
                st.warning(
                    f"No CARD reference genes tagged with an organism containing "
                    f"'{organism_input}'. Try a shorter/partial name, e.g. 'coli' instead of "
                    f"'Escherichia coli'."
                )
            else:
                st.write(f"Found **{len(matches)}** CARD reference gene(s) for '{organism_input}'")
                st.write("Documented drug classes (per CARD curation):")
                st.dataframe(drug_class_counts.rename("Count"), use_container_width=True)

                if cross_check_df is not None:
                    st.write("Cross-check against the trained model's own prediction:")
                    st.dataframe(cross_check_df, hide_index=True, use_container_width=True)
                else:
                    st.caption(
                        "None of these genes fall within the drug classes the model was "
                        "trained on, so no model cross-check is available."
                    )
