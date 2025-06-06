import spacy
import torch
import numpy as np
import pyphen
import string
from gensim.models.fasttext import load_facebook_model
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from collections import Counter
from nltk.corpus import stopwords
import nltk
from collections import Counter, deque

# Initialize models
nlp = spacy.load('de_dep_news_trf')
# fasttext_model = load_facebook_model('cc.de.300.bin')
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
bert_model = AutoModel.from_pretrained('bert-base-german-cased')
sentiment_tokenizer = AutoTokenizer.from_pretrained('oliverguhr/german-sentiment-bert')
sentiment_model = AutoModelForSequenceClassification.from_pretrained('oliverguhr/german-sentiment-bert')
dic = pyphen.Pyphen(lang='de')

# Compute BERT-based sentence coherence
def compute_bert_coherence(sentences):
    embeddings = []
    for sentence in sentences:
        inputs = bert_tokenizer(sentence.text, return_tensors='pt', truncation=True)
        outputs = bert_model(**inputs)
        # Use the [CLS] token representation
        embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(embedding[0])
    if len(embeddings) < 2:
        return None
    similarities = []
    for i in range(len(embeddings) - 1):
        vec1 = embeddings[i]
        vec2 = embeddings[i + 1]
        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        similarities.append(sim)
    return np.mean(similarities)

# Compute Readability
def compute_readability(doc):
    sentences = list(doc.sents)
    if not sentences:
        return None
    total_words = sum(len(sent) for sent in sentences)
    avg_sentence_length = total_words / len(sentences)
    total_syllables = sum(len(dic.inserted(token.text.lower()).split('-')) for token in doc if token.is_alpha)
    avg_syllables_per_word = total_syllables / total_words if total_words > 0 else 0
    readability = 180 - avg_sentence_length - (58.5 * avg_syllables_per_word)
    return readability

# Compute POS ratios
def compute_pos_ratios(tokens):
    pos_counts = Counter(token.pos_ for token in tokens)
    total_tokens = sum(pos_counts.values())
    ratios = {}
    # Ratios of specific POS tags to total tokens
    for pos in ['PRON', 'ADJ', 'ADV', 'DET', 'NOUN', 'VERB']:
        ratios[f'{pos}_Ratio'] = pos_counts.get(pos, 0) / total_tokens if total_tokens > 0 else 0
    # Total number of pronouns
    total_pronouns = pos_counts.get('PRON', 0)
    # Personal pronouns ratio to total pronouns
    pers_pron_count = sum(1 for token in tokens if token.pos_ == 'PRON' and 'Prs' in token.morph.get('PronType', []))
    ratios['Pers_PRON_Ratio'] = pers_pron_count / total_pronouns if total_pronouns > 0 else 0
    # Possessive pronouns ratio to total pronouns
    poss_pron_count = sum(1 for token in tokens if token.pos_ == 'PRON' and 'Poss' in token.morph.get('PronType', []))
    ratios['Poss_PRON_Ratio'] = poss_pron_count / total_pronouns if total_pronouns > 0 else 0
    return ratios


# Compute sentiment probability
def compute_negative_sentiment_probability(sentences):
    import torch
    negative_probs = []
    for sentence in sentences:
        inputs = sentiment_tokenizer(sentence.text, return_tensors='pt', truncation=True).to(sentiment_model.device)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        negative_probs.append(probs[0])  # Assuming negative class is at index 0
    if negative_probs:
        return np.mean(negative_probs)
    else:
        return None


# Function to compute Type-Token Ratio (TTR)
def compute_ttr(tokens):
    types = set(tokens)
    ttr = len(types) / len(tokens) if len(tokens) > 0 else 0
    return ttr

# Function to compute average sentence length
def compute_avg_sentence_length(doc):
    sentences = list(doc.sents)
    if len(sentences) == 0:
        return 0
    total_length = sum(len(sentence) for sentence in sentences)
    avg_length = total_length / len(sentences)
    return avg_length

# Function to count connectives
def compute_connective_ratio(tokens, num_sentences):
    # List of common German connectives
    connectives = set([
        'und', 'aber', 'oder', 'weil', 'denn', 'doch', 'jedoch', 'dass', 'damit', 'obwohl',
        'während', 'deshalb', 'trotzdem', 'sobald', 'sowie', 'sowohl', 'als', 'ob', 'bevor',
        'nachdem', 'falls', 'sofern', 'indem', 'dadurch', 'folglich', 'also'
    ])
    connective_count = sum(1 for token in tokens if token.text.lower() in connectives)
    connective_ratio = connective_count / num_sentences if num_sentences > 0 else 0
    return connective_ratio


# Function to compute morphological complexity
def compute_morphological_complexity(tokens):
    morph_complexity = np.mean([len(token.morph) for token in tokens if token.is_alpha])
    return morph_complexity


# Function to compute lexical diversity using MTLD (Measure of Textual Lexical Diversity)
def compute_mtld(tokens):
    from lexical_diversity import lex_div as ld
    words = [token.text.lower() for token in tokens if token.is_alpha]
    mtld = ld.mtld(words)
    return mtld

# Function to compute semantic density (global)
def compute_semantic_density(doc):
    # Compute the average pairwise cosine similarity between all content words
    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    word_vectors = [wv[word] for word in words if word in wv.key_to_index]
    if len(word_vectors) < 2:
        return None
    similarities = []
    for i in range(len(word_vectors)):
        for j in range(i + 1, len(word_vectors)):
            vec1 = word_vectors[i]
            vec2 = word_vectors[j]
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append(sim)
    semantic_density = np.mean(similarities)
    return semantic_density

# Function to compute syntactic complexity (e.g., average parse tree depth)
def compute_syntactic_complexity(doc):
    depths = []
    for sentence in doc.sents:
        depths.append(get_tree_depth(sentence.root))
    if depths:
        avg_depth = np.mean(depths)
        return avg_depth
    else:
        return None

# Helper function to compute tree depth
def get_tree_depth(token):
    if not list(token.children):
        return 1
    else:
        return 1 + max(get_tree_depth(child) for child in token.children)

# Function to compute semantic coherence (local)
def compute_semantic_coherence(doc):
    # Computes the average cosine similarity between consecutive sentences
    sentences = [sent for sent in doc.sents if len(sent) > 0]
    if len(sentences) < 2:
        return None  # Not enough sentences to compute coherence
    sentence_vectors = []
    for sentence in sentences:
        words = [token.text.lower() for token in sentence if token.is_alpha and not token.is_stop]
        word_vectors = [wv[word] for word in words if word in wv.key_to_index]
        if word_vectors:
            sentence_vector = np.mean(word_vectors, axis=0)
            sentence_vectors.append(sentence_vector)
        else:
            sentence_vectors.append(None)
    # Compute cosine similarities between consecutive sentence vectors
    similarities = []
    for i in range(len(sentence_vectors) - 1):
        vec1 = sentence_vectors[i]
        vec2 = sentence_vectors[i + 1]
        if vec1 is not None and vec2 is not None:
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities.append(sim)
    if similarities:
        avg_similarity = np.mean(similarities)
        return avg_similarity
    else:
        return None


# Count filled pauses and repetitions
def compute_disfluencies(tokens):
    # Common German filler words
    fillers = set(['aaa', 'ah', 'aah', 'ach', 'aha', 'ahm', 'eh', 'ehm', 'ha', 'haha', 'he', 'hehe', 'hm', 'hmm', 'hmmh', 'hmhm', 'hmmm', 'hmmmm',
    'hä', 'ja', 'jaa', 'joa', 'mh', 'mhh', 'mhh', 'mhm', 'mhmhmh', 'mhmhmhmh', 'mmh', 'mmm', 'mmmm', 'mmhhhmhm', 'mmhm', 'mmmhm',
    'naja', 'nja', 'och', 'oh', 'puh', 'puuuh', 'tja', 'uff' , 'uh', 'äh', 'äha', '#hh', 'ähm', 'öh', 'öhm'])
    filled_pauses = sum(1 for token in tokens if token.text.lower() in fillers)
    # Simple repetition detection (consecutive identical words)
    repetitions = sum(1 for i in range(1, len(tokens)) if tokens[i].text.lower() == tokens[i-1].text.lower())
    # Total words
    total_words = len([token for token in tokens if token.is_alpha])
    # Ratios
    filled_pauses_ratio = filled_pauses / total_words if total_words > 0 else 0
    repetitions_ratio = repetitions / total_words if total_words > 0 else 0
    return filled_pauses_ratio, repetitions_ratio

# Compute subordination index
def compute_subordination_index(doc):
    total_clauses = 0
    subordinate_clauses = 0

    for sent in doc.sents:
        total_clauses += 1  # Count the main clause
        for token in sent:
            if token.dep_ in ['oc', 'rc']:
                subordinate_clauses += 1
                total_clauses += 1  # Each subordinate clause is a clause
            elif token.dep_ == 'cp':
                # Look for the head of the complementizer, which is often the verb of the subordinate clause
                head_token = token.head
                if head_token.dep_ == 'oc':
                    continue  # Already counted
                else:
                    subordinate_clauses += 1
                    total_clauses += 1
    subordination_index = subordinate_clauses / total_clauses if total_clauses > 0 else 0
    return subordination_index


# Count grammatical errors
def compute_grammatical_errors(doc):
    errors = 0
    for token in doc:
        if token.pos_ == 'X':
            errors += 1
    total_tokens = len(doc)
    grammatical_error_ratio = errors / total_tokens if total_tokens > 0 else 0
    return grammatical_error_ratio

# 

def calculate_moving_average_ttr(doc, window_size=50):
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    if len(tokens) < window_size:
        return len(set(tokens)) / len(tokens) if tokens else 0
    type_counts = deque(maxlen=window_size)
    seen_types = set()
    moving_ttr = []
    for token in tokens:
        if token not in seen_types:
            seen_types.add(token)
            type_counts.append(1)
        else:
            type_counts.append(0)
        if len(type_counts) == window_size:
            moving_ttr.append(sum(type_counts) / window_size)
    return sum(moving_ttr) / len(moving_ttr) if moving_ttr else 0

def calculate_morphological_root_overlap(doc):
    sentence_roots = [set(token.lemma_ for token in sent if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']) for sent in doc.sents]
    overlap_count = 0
    total_pairs = 0
    for i in range(len(sentence_roots)):
        for j in range(i + 1, len(sentence_roots)):
            overlap_count += len(sentence_roots[i].intersection(sentence_roots[j]))
            total_pairs += 1
    return overlap_count / total_pairs if total_pairs else 0


# Function to calculate Mean Dependency Distance (MDD)
def calculate_mean_dependency_distance(doc):
    total_distance = sum(abs(token.i - token.head.i) for token in doc)
    total_dependencies = len([token for token in doc])
    return total_distance / total_dependencies if total_dependencies else 0


# Function to count simple sentences
def count_simple_sentences(doc):
    simple_sentences_count = 0
    for sent in doc.sents:
        if not any(token.dep_ == 'mark' or token.dep_ == 'cc' for token in sent):
            if not any(child.dep_ == 'cj' for token in sent for child in token.children):
                simple_sentences_count += 1
    return simple_sentences_count

