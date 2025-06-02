# AI-Generated Text Detection

<img src="https://github.com/JonMcEntee/ieee-cis-fraud-detection/blob/main/credit-card-fraud.png?raw=true" width="50%" height="50%">

This repository contains [a Jupyter Notebook](https://github.com/JonMcEntee/student-essay-ai-text-detection/blob/main/AI_generated_Text_Detection.ipynb) for detecting AI-generated text using classical and transformer-based NLP techniques. The project was developed as part of the [LLM - Detect AI-Generated Text Kaggle competition](https://www.kaggle.com/c/llm-detect-ai-generated-text).

## Overview

The notebook explores methods to distinguish between human-written and AI-generated text using:

- **Classical NLP techniques**: TF-IDF vectorization and logistic regression
- **Advanced transformer-based approaches**: Fine-tuning a DeBERTa model with contrastive learning

Key datasets used:
- Original competition dataset from Kaggle
- Community-augmented [DAIGT v2 dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)

## Key Features

1. **Exploratory Data Analysis (EDA)**
   - Analysis of text length distributions
   - Visualization of label distributions across different prompts
   - Examination of class imbalances

2. **Modeling Approaches**
   - Baseline TF-IDF + Logistic Regression pipeline
   - Advanced DeBERTa fine-tuning with contrastive learning
   - UMAP visualization of text embeddings

3. **Results**
   - Comparison of model performance
   - Visualization of embedding spaces
   - Interpretation of model behavior

## Requirements

The notebook requires the following Python packages:
- pandas, numpy
- scikit-learn
- transformers, sentence-transformers
- matplotlib, seaborn
- nltk
- umap-learn

Install with:
```bash
pip install -U transformers accelerate peft huggingface_hub pandas numpy scikit-learn matplotlib seaborn nltk umap-learn
```

## Usage

1. Mount your Google Drive (if running in Colab)
2. Load the datasets from the specified paths
3. Run the notebook cells sequentially to:
   - Perform EDA
   - Train baseline models
   - Fine-tune transformer models
   - Evaluate and visualize results

## Key Findings

- Transformer embeddings (especially contrastively trained) provide more discriminative features than classical approaches
- The DAIGT v2 dataset offers better prompt diversity and class balance than the original competition data
- Visualization techniques like UMAP help interpret model behavior and feature spaces

## Acknowledgments

- Kaggle for hosting the competition
- The DAIGT v2 dataset contributors
- Hugging Face for transformer models
