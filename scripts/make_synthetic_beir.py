#!/usr/bin/env python3
"""Generates a tiny BEIR-formatted dataset (corpus.jsonl, queries.jsonl,
qrels/{train,test}.tsv) for offline smoke-testing the pipeline. See
configs/synthetic_smoke.yaml. Deterministic given the seed.

This is not a stand-in for scifact's difficulty or scale. It exists so the
six stages can be exercised end to end with zero network access; treat any
metric it produces as "did the code run," not "does the method work."
"""
import argparse
import json
import os
import random

TOPICS = {
    "photosynthesis": ["chlorophyll", "sunlight", "carbon", "dioxide", "glucose", "leaf", "plant", "oxygen"],
    "black_holes": ["gravity", "event", "horizon", "singularity", "spacetime", "mass", "collapse", "star"],
    "vaccines": ["immune", "antibody", "pathogen", "dose", "trial", "antigen", "vaccination", "response"],
    "climate_change": ["emissions", "temperature", "carbon", "warming", "atmosphere", "ice", "sea", "level"],
    "neural_networks": ["weights", "gradient", "layer", "training", "backpropagation", "neuron", "activation", "loss"],
}
FILLER = ["study", "results", "analysis", "method", "data", "research", "found", "shows", "recent", "using"]


def make_doc(rng: random.Random, topic: str, on_topic: bool) -> str:
    """Generate a semi-coherent document with template-based structure.

    Neural encoders need actual semantic coherence, not random word salad.
    BM25 will match regardless, but cross-encoders trained on real text
    need something that looks like a sentence.
    """
    vocab = TOPICS[topic]
    if on_topic:
        # Pick 2-3 topic-specific terms to anchor the sentence
        terms = rng.sample(vocab, k=min(3, len(vocab)))
        templates = [
            f"Recent research shows that {terms[0]} plays a key role in {topic.replace('_', ' ')}, alongside {terms[1]}.",
            f"Studies found that {terms[0]} and {terms[1]} are critical for understanding {topic.replace('_', ' ')}.",
            f"Analysis of {terms[0]} reveals important insights about {terms[1]} in {topic.replace('_', ' ')} research.",
            f"Data indicates {terms[0]} interacts with {terms[1]} during {topic.replace('_', ' ')} processes.",
        ]
        return rng.choice(templates)
    else:
        # Off-topic distractors: generic filler without topic terms
        filler = rng.sample(FILLER, k=4)
        return f"The {filler[0]} examined various {filler[1]} and {filler[2]} using standard {filler[3]} methods."


def build(seed: int = 42, docs_per_topic: int = 9, queries_per_topic: int = 5) -> dict:
    rng = random.Random(seed)
    corpus, queries, qrels_train, qrels_test = {}, {}, {}, {}
    doc_id = 0

    topic_docs = {t: [] for t in TOPICS}
    for topic in TOPICS:
        for i in range(docs_per_topic):
            on_topic = i < docs_per_topic - 2  # last two per topic are near-miss distractors
            doc_id += 1
            did = f"d{doc_id}"
            text = make_doc(rng, topic, on_topic)
            corpus[did] = {"title": f"{topic.replace('_', ' ')} note {i}", "text": text}
            topic_docs[topic].append((did, on_topic))

    query_id = 0
    for topic, vocab in TOPICS.items():
        for i in range(queries_per_topic):
            query_id += 1
            qid = f"q{query_id}"
            terms = rng.sample(vocab, k=3)
            queries[qid] = f"what is the role of {' '.join(terms)} in {topic.replace('_', ' ')}"
            relevant = {did: (2 if j == 0 else 1) for j, (did, on_topic) in enumerate(topic_docs[topic]) if on_topic}
            target = qrels_train if i < queries_per_topic - 2 else qrels_test  # last 2/topic held out for test
            target[qid] = relevant

    return {"corpus": corpus, "queries": queries, "qrels_train": qrels_train, "qrels_test": qrels_test}


def write(dataset: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "qrels"), exist_ok=True)

    with open(os.path.join(out_dir, "corpus.jsonl"), "w") as f:
        for did, doc in dataset["corpus"].items():
            f.write(json.dumps({"_id": did, "title": doc["title"], "text": doc["text"]}) + "\n")

    with open(os.path.join(out_dir, "queries.jsonl"), "w") as f:
        for qid, text in dataset["queries"].items():
            f.write(json.dumps({"_id": qid, "text": text}) + "\n")

    for split, qrels in (("train", dataset["qrels_train"]), ("test", dataset["qrels_test"])):
        with open(os.path.join(out_dir, "qrels", f"{split}.tsv"), "w") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for qid, docs in qrels.items():
                for did, score in docs.items():
                    f.write(f"{qid}\t{did}\t{score}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./data/beir/synthetic_demo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write(build(args.seed), args.out)
    print(f"wrote synthetic BEIR dataset to {args.out}")
