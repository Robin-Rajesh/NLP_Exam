# 🎬 Movie Review Spoiler Detection

An NLP project that detects spoilers in IMDB movie reviews, comparing a classical machine learning model with a deep learning model.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD)
![TensorFlow](https://img.shields.io/badge/Keras-BiLSTM-D00000?logo=keras&logoColor=white)

## 🧪 Approach

1. **Data**: the [IMDB 50K movie reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
2. **Heuristic labeling**: reviews are labeled as spoilers if they contain phrases such as *"turns out"*, *"in the end"* or *"was actually"*.
3. **Balancing**: the majority class is downsampled so spoiler and non-spoiler reviews are equal.
4. **Model 1, TF-IDF + XGBoost**: 1–3 word n-grams, up to 10,000 features.
5. **Model 2, BiLSTM + GloVe**: frozen 100-dimensional GloVe embeddings feeding a bidirectional LSTM.
6. **Evaluation**: classification reports for both models, plus class distributions and the top XGBoost feature importances.

## 📂 Files

- `NLP_Exam (3).ipynb`: the notebook (written for Google Colab)
- `NLP_Exam.pdf`: the exam write-up

## 🚀 Running

Open the notebook in Google Colab and upload `IMDB Dataset.csv` and `glove.6B.100d.txt` (from [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)) to `/content/`.
