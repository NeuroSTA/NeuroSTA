import os
import pandas as pd
import spacy
import re

# Load the transformer-based German language model
nlp = spacy.load("de_dep_news_trf")

# Define conjunction lists
coordinating_conjunctions = [
    "und", "oder", "aber", "denn", "sondern", "doch", "jedoch", "sowohl", "weder", "entweder"
]
subordinating_conjunctions = [
    "bevor", "bis", "da", "damit", "dass", "ehe", "falls", "indem", "indes", "indessen",
    "obgleich", "obschon", "obwohl", "seit", "seitdem", "sobald", "solange", "so dass",
    "sodass", "soweit", "sowie", "sofern", "soviel", "weil", "wenn", "wie", "wobei", "während",
    "wo", "wohingegen", "zumal"
]


# Function to count total words in a file
def count_words_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        words = text.split()
        return len(words)

# Function to count unique words and calculate Type-Token Ratio (TTR)
def count_unique_words_and_ttr(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        doc = nlp(text)
        words = [token.text.lower() for token in doc if token.is_alpha]
        unique_words = len(set(words))
        total_words = len(words)
        ttr = unique_words / total_words if total_words > 0 else 0
        return unique_words, ttr

# Function to count sentences in a file
def count_sentences_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        sentences = re.split(r'[.!?]', text)
        return len([s.strip() for s in sentences if s.strip()])

# Function to calculate Mean Length of Utterance (MLU)
def calculate_mlu(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        sentences = re.split(r'[.!?]', text)
        words = [sentence.split() for sentence in sentences]
        total_words = sum(len(sentence) for sentence in words)
        total_sentences = len(sentences)
        return total_words / total_sentences if total_sentences > 0 else 0

# Function to calculate MLU excluding filler words
def calculate_mlu_no_fillers(file_path):
    fillers = {"äh", "ähm", "hm", "mh", "eh", "mhm", "öh", "öhm"}
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        sentences = re.split(r'[.!?]', text)
        words = [[word for word in sentence.split() if word.lower() not in fillers] for sentence in sentences]
        total_words = sum(len(sentence) for sentence in words)
        total_sentences = len(sentences)
        return total_words / total_sentences if total_sentences > 0 else 0

# Function to calculate Noun-Verb Ratio (NVR)
def calculate_noun_verb_ratio(text):
    doc = nlp(text)
    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    return noun_count / verb_count if verb_count > 0 else 0
# Function to calculate Open-Closed Ratio (OCR)
def calculate_open_closed_ratio(text):
    doc = nlp(text)
    content_words = sum(1 for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"])
    function_words = sum(1 for token in doc if token.pos_ in ["DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "INTJ"])
    return content_words / function_words if function_words > 0 else 0

def count_simple_sentences(text):
    doc = nlp(text)
    return sum(1 for sent in doc.sents if not any(token.dep_ in ["mark", "cc"] for token in sent))

# Function to count coordinating and subordinating conjunctions
# This function counts how many times a given set of conjunctions appear in a text.

def count_conjunctions(text, conjunctions):
    doc = nlp(text)
    return sum(1 for token in doc if token.text.lower() in conjunctions)
