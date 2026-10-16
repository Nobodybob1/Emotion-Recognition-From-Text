# Emotion Recognition From Text

A from-scratch PyTorch classifier that reads a sentence and predicts which of six emotions it expresses: **anger, fear, joy, love, sadness, surprise**.

The model is a 2-layer bidirectional LSTM with attention pooling, trained on the Kaggle [Emotions dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp). It reaches **93.0% accuracy** and **0.90 macro-F1** on the test set.

```python
predict_emotion("I am afraid of the dark.")  # -> 'fear'
```

## Repository contents

| File | Description |
|------|-------------|
| `main.ipynb` | The full pipeline: data loading, tokenization, model, training, evaluation, inference |
| `best_model.pth` | Trained weights (best checkpoint by validation macro-F1) |
| `requirements.txt` | Python dependencies |

## Setup

1. **Install dependencies** (Python 3.9+):

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   For GPU training, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before running the command above. The notebook uses the GPU automatically if one is available and falls back to the CPU otherwise.

2. **Download the dataset.** The data is not in the repo. Download it from [Kaggle](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) and put the files in an `archive/` folder:

   ```
   archive/
   ├── train.txt
   ├── val.txt
   └── test.txt
   ```

   Each line has the form `text;label`, e.g. `i feel so happy today;joy`.

   > The dataset is needed even if you only want to use the saved model, because the vocabulary is rebuilt from `train.txt` on every run.

3. **Open the notebook:**

   ```bash
   jupyter notebook main.ipynb
   ```

## Usage

- **Use the pretrained model:** run every cell **except the training loop** (cell 8). Cell 9 loads `best_model.pth`.
- **Retrain:** run all cells. Training overwrites `best_model.pth` with the new best checkpoint.
- **Predict your own text:** after the model is loaded, call `predict_emotion("your sentence")`.

## Data

| Split | Examples |
|-------|----------|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

The classes are imbalanced. Training counts: joy 5,362 · sadness 4,666 · anger 2,159 · fear 1,937 · love 1,304 · surprise 572.

## How it works

**Preprocessing**
- Lowercase the text and tokenize with a regex (words, numbers and punctuation become separate tokens).
- Build the vocabulary from the training set (~15k tokens, capped at 20k), plus `<pad>` and `<unk>`.
- Pad or truncate every sentence to 60 tokens.

**Model: `BiLSTMClassifier`**
```
Embedding (128) → Dropout
  → BiLSTM (2 layers, 256 hidden per direction, dropout between layers)
  → Attention pooling (padding masked out)
  → Dropout → Linear → 6 classes
```
Attention pooling learns a weight for each token and takes a weighted average of the LSTM outputs, so the model can focus on emotionally important words instead of relying only on the final hidden state.

**Training**
- Loss: cross-entropy with inverse-frequency class weights, to counter class imbalance
- Optimizer: Adam (lr `1e-3`, weight decay `1e-4`), batch size 64
- Scheduler: `ReduceLROnPlateau` on validation macro-F1 (factor 0.5, patience 2)
- Early stopping: patience 4, up to 15 epochs
- The best checkpoint by validation macro-F1 (0.914) is saved
- Seed fixed at 42

## Results (test set)

| Metric | Score |
|--------|-------|
| Accuracy | **0.930** |
| Macro-F1 | **0.900** |

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| anger | 0.92 | 0.96 | 0.94 |
| fear | 0.91 | 0.88 | 0.90 |
| joy | 0.99 | 0.90 | 0.94 |
| love | 0.74 | 0.99 | 0.84 |
| sadness | 0.99 | 0.95 | 0.97 |
| surprise | 0.68 | 1.00 | 0.81 |

**Observations**
- Sadness, joy and anger are classified very reliably.
- The rare classes, **love** and **surprise**, have near-perfect recall but lower precision. The inverse-frequency class weights push the model to over-predict them.
- The most common mistakes are *joy → love* (54 cases) and *fear → surprise* (23 cases). These emotions are semantically close.

## Future work

- Soften the class weights (e.g. `1/sqrt(count)`) to balance precision and recall on the rare classes
- Transfer learning: fine-tune a pretrained transformer (e.g. DistilBERT) and compare it against this baseline
- Save the vocabulary with the model, so inference doesn't require the training data
