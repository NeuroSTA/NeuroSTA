import spacy
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load('de_core_news_lg')


def calculate_semantic_coherence(doc):
    """
    Calculates the semantic coherence of a document by computing the average cosine similarity
    between the word vectors of content words within the document.
    """
    vectors = [token.vector for token in doc if token.has_vector and not token.is_stop]
    if len(vectors) < 2:
        return 0
    # Compute pairwise cosine similarity between word vectors and return the mean
    similarity = cosine_similarity(vectors)
    return np.mean(similarity)


def calculate_syntactic_complexity(doc):
    """
    Calculates the syntactic complexity of a document based on sentence length and
    dependency tree depth for each sentence.
    """
    sentence_lengths = [len(sent) for sent in doc.sents]
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0

    # Calculate the depth of dependency trees
    tree_depths = []
    for sent in doc.sents:
        depths = [token.head.i - token.i for token in sent]  # Use distance from head for depth
        if depths:
            tree_depths.append(max(depths))
    avg_tree_depth = np.mean(tree_depths) if tree_depths else 0

    # Combine avg_sentence_length and avg_tree_depth into a syntactic complexity score
    syntactic_complexity_score = avg_sentence_length * avg_tree_depth

    return syntactic_complexity_score, avg_sentence_length, avg_tree_depth


def build_speech_graph(doc):
    """
    Builds a speech graph where nodes represent words and edges connect consecutive words.
    Calculates the average shortest path length as a measure of graph coherence.
    """
    words = [token.text for token in doc if not token.is_punct and not token.is_stop]
    graph = nx.Graph()

    # Add edges between consecutive words
    for i in range(len(words) - 1):
        graph.add_edge(words[i], words[i + 1])

    try:
        # Calculate the average shortest path length in the graph
        return nx.average_shortest_path_length(graph)
    except nx.NetworkXError:
        # If the graph is too small or disconnected, return 0
        return 0
