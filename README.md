# NeuroSTA

**Neuroimaging and Speech using Transdiagnostic Approaches**

NeuroSTA brings together computational approaches to investigate speech, language, cognition, and brain structure in psychiatric research. Our work emphasizes **transdiagnostic perspectives**, examining patterns and dimensions of variation that extend beyond conventional diagnostic categories.

This repository provides research code for linguistic and acoustic feature extraction, statistical modeling, machine learning, network analysis, and neuroimaging-based subtyping. The aim is to support **transparent, reproducible, and reusable computational research** at the intersection of neuroscience, psychiatry, and data science.

## Research areas

The repository is organized into independent analysis modules. Each research area includes its own documentation, methodological details and instructions for use.

| Research area | Description |
| --- | --- |
| **[Speech Feature Extraction](Speech_Features_Extraction/)** | Computational characterization of spoken language, including [acoustic and prosodic features](Speech_Features_Extraction/Acoustic_Features/README.md), [semantic and syntactic measures](Speech_Features_Extraction/Semantics-Syntax/README.md), and [speech-graph analysis](Speech_Features_Extraction/SpeechGraph/README.md). |
| **[NLP Speech Factors](NLP_Speech_Factors/README.md)** | Analysis of linguistic features and their underlying dimensions using natural language processing and exploratory factor analysis. |
| **[Machine Learning Classification](ML_Classification/README.md)** | Binary and multiclass classification workflows, feature-set comparisons, and approaches to combining information across modalities. |
| **[Network Analysis](Network_Analysis/README.md)** | Statistical network modeling to examine associations and interrelationships among psychological, cognitive, and clinical variables. |
| **[Transdiagnostic SuStaIn](Transdiagnostic_SuStaIn/README.md)** | Structural MRI-based **Subtype and Stage Inference (SuStaIn)** to investigate heterogeneity and patterns of neuroanatomical variation across psychiatric diagnoses. |
| **[Cognition](Cognition/)** | Research materials related to cognitive data analysis, including regression, dimensionality reduction, and clustering. |

## Repository structure and usage

Each module is maintained as a separate research workflow rather than as part of a single software package. For implementation details, required dependencies, input data formats, and execution instructions, please refer to the respective module's README.

To obtain the repository:

```bash
git clone https://github.com/NeuroSTA/NeuroSTA.git
cd NeuroSTA
```

Code may require dataset-specific configuration and access to external software, pretrained models, or computing infrastructure.

## Open science and reproducibility

NeuroSTA aims to facilitate transparent and reproducible computational research by sharing analysis code and methodological documentation. 

## Citation

If you use code from this repository, please cite the relevant associated publication, where applicable, and reference the repository and the specific commit or version used. Publication and citation details may also be provided in individual modules.

## Contact and contributions

Questions, suggestions, and contributions are welcome via [GitHub Issues](https://github.com/NeuroSTA/NeuroSTA/issues). 
For more information about NeuroSTA, our research, team, and publications, please visit the [NeuroSTA website](https://neurosta.github.io/NeuroSTA/).

