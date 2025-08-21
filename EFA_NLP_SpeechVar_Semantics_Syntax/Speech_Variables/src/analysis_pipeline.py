import os
import pandas as pd
from linguistics import (compute_ttr, compute_mtld, compute_connective_ratio, compute_morphological_complexity,
                         compute_semantic_coherence, compute_semantic_density, compute_readability,
                         compute_bert_coherence, compute_pos_ratios, compute_negative_sentiment_probability,
                         compute_disfluencies, compute_subordination_index, compute_grammatical_errors)
from graph_analysis import calculate_semantic_coherence, calculate_syntactic_complexity, build_speech_graph
import spacy

TRANSCRIPT_FOLDER = '../data/transcripts/'


def run_linguistic_analysis():
    results = []
    nlp = spacy.load('de_dep_news_trf')
    for filename in os.listdir(TRANSCRIPT_FOLDER):
        filepath = os.path.join(TRANSCRIPT_FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()

        doc = nlp(text)
        tokens = [token for token in doc if not token.is_punct and not token.is_space]
        sentences = list(doc.sents)

        # Compute linguistic features
        result = {
            'Filename': filename,
            'BERT_Coherence': compute_bert_coherence(sentences),
            'Readability': compute_readability(doc),
            'POS_Ratios': compute_pos_ratios(tokens),
            'Negative_Sentiment': compute_negative_sentiment_probability(sentences),
            'TTR': compute_ttr([token.text.lower() for token in tokens if token.is_alpha]),
            'MTLD': compute_mtld(tokens),
            'Morph_Complexity': compute_morphological_complexity(tokens),
            'Disfluencies': compute_disfluencies(tokens),
            'Subordination_Index': compute_subordination_index(doc),
            'Grammatical_Errors': compute_grammatical_errors(doc)
        }
        results.append(result)

    df = pd.DataFrame(results)
    df.to_excel('../results/linguistics_results.xlsx', index=False)


def run_graph_analysis():
    results = []
    nlp = spacy.load('de_core_news_lg')  # Ensuring correct model in each analysis
    for filename in os.listdir(TRANSCRIPT_FOLDER):
        filepath = os.path.join(TRANSCRIPT_FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()

        doc = nlp(text)

        result = {
            'Filename': filename,
            'Semantic_Coherence': calculate_semantic_coherence(doc),
            'Syntactic_Complexity': calculate_syntactic_complexity(doc)[0],
            'Speech_Graph_Coherence': build_speech_graph(doc)
        }
        results.append(result)

    df = pd.DataFrame(results)
    df.to_excel('../results/graph_results.xlsx', index=False)


if __name__ == "__main__":
    run_linguistic_analysis()
    run_graph_analysis()