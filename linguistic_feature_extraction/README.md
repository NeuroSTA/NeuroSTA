# Speech Analysis Pipeline

This project analyzes German speech transcripts, extracting linguistic features.


## Installation and Setup

### Prerequisites
1. **Python 3.12.4**
2. **Virtual Environment (recommended)**

### Step-by-Step Setup
1. **Clone the Repository**
    <!-- ```bash
    git clone https://github.com/svenjaseuffert/NLP
    cd Speech_analysis
    ``` -->

2. **Set Up a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**
    pip install -r requirements.txt


4. **Download Necessary NLP Models**
   - **spaCy Models**:
     ```bash
     python -m spacy download de_dep_news_trf
     python -m spacy download de_core_news_lg
     ```

5. **Download NLTK Stopwords**:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## Usage

1. **Prepare Input Data**: Place `.txt` transcript files in the `data/transcripts/` directory.
2. **Run the Analysis Pipeline**:
    ```bash
    python src/analysis_pipeline.py
    ```
3. **Merge Results**:
    ```bash
    python src/merge_results.py
    ```
4. **View Results**:
   - Linguistic results are saved as `combined_analysis.xlsx` in the `results` folder.
   
## Output

The analysis output provides the following metrics:

### Linguistic Features
- **Type-Token Ratio (TTR)**: Lexical diversity.
- **MTLD**: Measure of Textual Lexical Diversity.
- **POS Ratios**: Ratios of parts of speech like pronouns, nouns, verbs, etc.
- **Morphological Complexity**: Average morphological feature count per word.
- **Syntactic Complexity**: Composite metric of sentence length and dependency tree depth.
- **Readability Index**: German-specific readability metric.
- **Subordination Index**: Ratio of subordinate clauses to main clauses.
- **Grammatical Error Ratio**: Count of tokens with potential grammatical errors.
- **Semantic Density**: Average cosine similarity between word vectors (global).
- **Semantic Coherence**: Average cosine similarity between word vectors.
- **spacy Coherence**: Semantic coherence based on spacy word vectors across words
- **BERT Coherence**: Sentence coherence using BERT embeddings.
- **Connective Ratio**: Proportion of connective words to sentences.
- **Speech Graph Coherence**: Average shortest path length in the speech graph.
- **Disfluencies**: Ratios of filled pauses and word repetitions.
- **Negative Sentiment**: Probability of negative sentiment.

