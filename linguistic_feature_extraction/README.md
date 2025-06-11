
# Linguistic Analysis Pipeline

This project implements a comprehensive pipeline for analyzing German-language transcripts using advanced NLP methods. It extracts semantic and syntactic features.

## 🧪 Methodology

The pipeline performs multi-layered text analysis:

- **Preprocessing**: Normalizes contractions and cleans input text
- **Tokenization and Utterance Segmentation**: Based on spaCy German models
- **Feature Extraction**:
  - **Lexical**: TTR, MTLD, MATTR
  - **Syntactic**: Tree depth, subordination, sentence length
  - **Morphological**: Morph. complexity, root overlap
  - **Semantic**: Embedding-based coherence using Word2Vec, SBERT, BERT, XLM-R
  - **Graph-based**: Speech coherence graph metrics
  - **Discourse and Pragmatics**: Filler detection, sentiment

## 📁 Repository Structure

- `analysis_pipeline.py`: Full pipeline combining all modules
- `linguistics.py`: Linguistic feature computation
- `graph_analysis.py`: Syntactic/semantic graph metrics
- `LLP_features.py`: Lexico-structural metrics
- `merge_results.py`: Combines result tables
- `README.md`: Documentation
- `requirements.txt`: Software dependencies
- `__init__.py`: init for packaging

## 🗂 Data Input

Place plain `.txt` transcript files into:
```
data/transcripts/
```
Each file should be a UTF-8 encoded German language transcript.

## 🧠 Models Used

- `spaCy`: `de_dep_news_trf`, `de_core_news_lg`
- `Gensim`: Word2Vec (.bin format)
- `sentence-transformers`: SBERT multilingual
- `transformers`:
  - `bert-base-german-dbmdz-uncased`
  - `xlm-roberta-base`
  - `german-sentiment-bert`

## ▶ Usage

Install dependencies:
```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m spacy download de_dep_news_trf
python -m spacy download de_core_news_lg
```

Download pretrained embeddings and save into the models/ folder:

German Word2Vec: download from https://devmount.github.io/GermanWordEmbeddings/ and save as models/german.model

FastText (cc.de.300): download from https://fasttext.cc/docs/en/pretrained-vectors.html and save as models/cc.de.300.bin



Run the full pipeline:
```bash
python analysis_pipeline.py
```
(Optional) Merge with other results:
```bash
python merge_results.py
```

## 📝 Output

Results are saved as an Excel file:
```
results/combined_analysis.xlsx
```


## 📌 Reproducibility Notes

- Tested on Python 3.11 with sufficient RAM for transformer models
- Supports GPU for faster BERT/SBERT inference
- Pretrained models auto-downloaded via `transformers`

