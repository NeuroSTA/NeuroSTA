import os
import spacy
import re
import logging
import pandas as pd
import numpy as np
import os
import re
import spacy
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from flair.embeddings import TransformerDocumentEmbeddings
import flair
import logging
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, XLMRobertaTokenizer, XLMRobertaModel
from sklearn.metrics.pairwise import cosine_similarity

from linguistics import (
    compute_ttr, compute_mtld, compute_connective_ratio, compute_morphological_complexity,
    compute_semantic_coherence, compute_semantic_density, compute_readability,
    compute_bert_coherence, compute_pos_ratios, compute_negative_sentiment_probability,
    compute_disfluencies, compute_subordination_index, compute_grammatical_errors,
    calculate_mean_dependency_distance, calculate_morphological_root_overlap, calculate_moving_average_ttr,
    compute_avg_sentence_length, compute_syntactic_complexity, count_simple_sentences
)

from graph_analysis import calculate_semantic_coherence, calculate_syntactic_complexity, build_speech_graph
from rieke_features import (
    count_words_in_file, count_unique_words_and_ttr, calculate_mlu,
    calculate_mlu_no_fillers, calculate_noun_verb_ratio, calculate_open_closed_ratio,
    count_conjunctions, count_sentences_in_file, coordinating_conjunctions,
    subordinating_conjunctions
)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRANSCRIPT_FOLDER = 'combined_code/data/transcripts'
OUTPUT_FILE = 'combined_code/results/combined_analysis.xlsx'


contractions = {
    r'\bne\b': 'eine',
    r'\b\'n\b': 'ein',
    r'\bnen\b': 'einen',
    r'\bnem\b': 'einem',
    r'\bner\b': 'einer',
    r'\baufm\b': 'auf einem',
    r'\bauf\'m\b': 'auf einem',
    r'\birgendne\b': 'irgendeine',
    r'\bhat\'n\b': 'hat ein',
    r'\bdadrunter\b': 'darunter'
}


# Load spaCy model for German
nlp = spacy.load("de_dep_news_trf")

# Load pre-trained Word2Vec model using gensim
word2vec_model_path = 'combined_code/german.model'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)



# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-german-dbmdz-uncased')
bert_model = BertModel.from_pretrained('bert-base-german-dbmdz-uncased')

# Load SentenceTransformer model
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
sbert_model = SentenceTransformer(model_name)

# Load XLM-RoBERTa model and tokenizer
xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
xlmroberta_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')




# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    # Correct contractions
    for contraction, correction in contractions.items():
        text = re.sub(contraction, correction, text, flags=re.IGNORECASE)
    # Strip excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove numbers and non-alphabetic characters
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-ZäöüÄÖÜß\s]', '', text)
    return text


# Function to tokenize text
def tokenize_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

# Function to segment text into utterances
def segment_into_utterances(text):
    doc = nlp(text)
    utterances = [sent.text for sent in doc.sents]
    return utterances

# Encoding functions
def encode_with_model(tokens, model):
    return [model[token] for token in tokens if token in model]

def encode_with_bert(utterances):
    embeddings = []
    for utterance in utterances:
        inputs = tokenizer(utterance, return_tensors='pt', padding=True, truncation=True)
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

def encode_with_sbert(utterances):
    embeddings = sbert_model.encode(utterances)
    return embeddings

def encode_with_xlmroberta(utterances):
    embeddings = []
    for utterance in utterances:
        inputs = xlmroberta_tokenizer(utterance, return_tensors='pt', padding=True, truncation=True)
        outputs = xlmroberta_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)
    return np.vstack(embeddings)

# Function to calculate statistics for cosine similarities
def calculate_cosine_similarity_statistics(embeddings):
    if len(embeddings) < 2:
        return {stat: 0 for stat in ['mean', 'median', 'min', 'max', 'std', 'iqr', '90th_percentile']}

    cosine_similarities = [cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0] for i in
                           range(len(embeddings) - 1)]
    cosine_similarities = np.array(cosine_similarities)
    return {
        'mean': np.mean(cosine_similarities),
        'median': np.median(cosine_similarities),
        'min': np.min(cosine_similarities),
        'max': np.max(cosine_similarities),
        'std': np.std(cosine_similarities),
        'iqr': np.percentile(cosine_similarities, 75) - np.percentile(cosine_similarities, 25),
        '90th_percentile': np.percentile(cosine_similarities, 90)
    }


def run_linguistic_analysis():
    results = []
    nlp = spacy.load('de_dep_news_trf')
    print("Current working directory:", os.getcwd())  # Debugging

    for filename in os.listdir(TRANSCRIPT_FOLDER):
        filepath = os.path.join(TRANSCRIPT_FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        proband=filename[:6]
        print(proband)
        doc = nlp(text)
        tokens = [token for token in doc if not token.is_punct and not token.is_space]
        sentences = list(doc.sents)
        coord_conj_count = count_conjunctions(text, coordinating_conjunctions)
        subord_conj_count = count_conjunctions(text, subordinating_conjunctions)
        total_word_count = len([token for token in doc if token.is_alpha])
        total_sentences = len(list(doc.sents))
        mlu = total_word_count / total_sentences
        total_types = len(set(token.text.lower() for token in doc if token.is_alpha))
        simple_sentences = count_simple_sentences(doc)
        simple_sentences_ratio = simple_sentences / total_sentences
        preprocessed_txt = preprocess_text(text)
        tokensnew = tokenize_text(preprocessed_txt)
        utterances = segment_into_utterances(preprocessed_txt)
        word2vec_embeddings = encode_with_model(tokensnew, word2vec_model)
        # fasttext_embeddings = encode_with_model(tokens, fasttext_model)
        bert_embeddings = encode_with_bert(utterances)
        sbert_embeddings = encode_with_sbert(utterances)
        xlmroberta_embeddings = encode_with_xlmroberta(utterances)

        # Compute linguistic features
        result = {
            'Proband': proband,
            'BERT_Coherence': compute_bert_coherence(sentences),
            'Readability': compute_readability(doc),
            'POS_Ratios': compute_pos_ratios(tokens),
            'PRON_Ratio': list(compute_pos_ratios(tokens).values())[0], 
            'ADJ_Ratio':list(compute_pos_ratios(tokens).values())[1], 
            'ADV_Ratio':list(compute_pos_ratios(tokens).values())[2], 
            'DET_Ratio':list(compute_pos_ratios(tokens).values())[3], 
            'NOUN_Ratio':list(compute_pos_ratios(tokens).values())[4], 
            'VERB_Ratio':list(compute_pos_ratios(tokens).values())[5], 
            'Pers_PRON_Ratio':list(compute_pos_ratios(tokens).values())[6] ,
            'Poss_PRON_Ratio': list(compute_pos_ratios(tokens).values())[7],           
            'Negative_Sentiment': compute_negative_sentiment_probability(sentences),
            'TTR': compute_ttr([token.text.lower() for token in tokens if token.is_alpha]),
            'MTLD': compute_mtld(tokens),
            'Morph_Complexity': compute_morphological_complexity(tokens),
            'Disfluencies': compute_disfluencies(tokens),
            'Subordination_Index': compute_subordination_index(doc),
            'Grammatical_Errors': compute_grammatical_errors(doc),
            # sveja more
            'Mean_Dependency_Distance': calculate_mean_dependency_distance(doc),   
            'MATTR':calculate_moving_average_ttr(doc),      
            'Root_overlap':   calculate_morphological_root_overlap(doc),
            'Mean_Dependency_Distance':calculate_mean_dependency_distance(doc),
            'MLU': mlu,
            'total_types':total_types,
            'SimS':simple_sentences_ratio,
            # rieke
            'Total Words':count_words_in_file(filepath),
            "Unique Words":count_unique_words_and_ttr(filepath)[0],
            "Type-Token Ratio":count_unique_words_and_ttr(filepath)[1],
            "MLU with Fillers":calculate_mlu(filepath),
            "MLU without Fillers":calculate_mlu_no_fillers(filepath), 
            "Noun-Verb Ratio":calculate_noun_verb_ratio(text),
            "Open-Closed Ratio":calculate_open_closed_ratio(text),
            "Total Sentences":count_sentences_in_file(filepath),
            "Simple Sentences Count":simple_sentences,
            "Coordinating Conjunctions":coord_conj_count,
            "Subordinating Conjunctions":subord_conj_count,
            "Total Conjunctions":coord_conj_count + subord_conj_count,
            'Semantic_Coherence': calculate_semantic_coherence(doc),
            'Syntactic_Complexity': calculate_syntactic_complexity(doc)[0],
            'Speech_Graph_Coherence': build_speech_graph(doc)


        }

        for model_name, emb in zip(
                    ["Word2Vec", "BERT", "SBERT", "XLMR"],
                    [word2vec_embeddings, bert_embeddings, sbert_embeddings, xlmroberta_embeddings,
                     ]
                ):
                stats = calculate_cosine_similarity_statistics(emb)
                result.update({f"{model_name}_{k}": v for k, v in stats.items()})
        results.append(result)

    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_FILE, index=False)
    logging.info("Full analysis complete. Results saved.")




if __name__ == "__main__":
    run_linguistic_analysis()











