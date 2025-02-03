# Fake News Detection

## Author
**Liantsoa Randrianasimbolarivelo**  
January 2025

## Dataset Summary
The Fake News Classification Dataset contains over 45,000 unique English-language news articles. These articles are classified as **true (1)** or **false (0)**, providing a valuable resource for fake news detection research using Transformer models.

## Exploratory Data Analysis (EDA)

### Dataset Structure
- **Total records:** 72,134 rows, 4 columns.

### Data Types
| Column Name  | Data Type                      |
|-------------|--------------------------------|
| `Unnamed: 0` | int64 (index)                 |
| `title`      | object (text)                  |
| `text`       | object (text)                  |
| `label`      | int64 (binary classification) |

### Data Cleaning
- Removed the `Unnamed: 0` column (irrelevant index).
- Checked for missing values:
  - `title`: 558 missing values.
  - `text`: 39 missing values.
  - `label`: 0 missing values.
- Checked for duplicated records: 8,416 duplicates found and removed.

### Text Length Statistics
|  | Word Count | Character Count |
|----|------------|----------------|
| Count | 63,121 | 63,121 |
| Mean | 341.16 | 2451.03 |
| Standard Deviation | 369.81 | 2604.85 |
| Min | 2 | 11 |
| 25th Percentile | 158 | 1131 |
| Median | 256 | 1857 |
| 75th Percentile | 422 | 3055 |
| Max | 20,731 | 137,970 |

### Target Variable Distribution
![Proportion of True vs. Fake News](trainTarget.png)

### Most Frequent Words
![Most Frequent Words](WordMostFrequency.png)

## Preprocessing
- Merged `title` and `text` columns into a single `news` variable.
- Removed punctuation and special characters.
- Converted text to lowercase.
- Removed stopwords (e.g., "the", "is", "and").
- Applied tokenization and lemmatization.
- Applied tokenizer transformation sequences to text.
- Applied padding to ensure uniform input size.

## Modelization
The model uses a combination of word embeddings, Bidirectional Long Short-Term Memory (BiLSTM) layers, and regularization techniques to classify news articles as either fake or real. The model is implemented using TensorFlow and Keras.

### Model Architecture

#### Embedding Layer
The first layer is an **Embedding** layer, which maps input words into dense vectors of fixed size.
- **Vocabulary size:** 10,000 words.
- **Embedding dimensions:** 20.
- **Maximum input sequence length:** 200 tokens.

#### Bidirectional LSTM Layers
Two BiLSTM layers capture contextual information from text in both forward and backward directions.
- **First BiLSTM layer:** 64 units, returns sequences.
- **Second BiLSTM layer:** 32 units, returns only final output.
- Both layers use **L2 regularization (0.01)** to prevent overfitting.

#### Batch Normalization and Dropout
- **BatchNormalization**: Normalizes activations for stable training.
- **Dropout**: 50% of the units are randomly dropped during training.

#### Dense Layers
- **First dense layer:** 16 units, ReLU activation.
- **Output layer:** 1 unit, Sigmoid activation (for binary classification).

#### Model Compilation
- **Loss function:** Binary cross-entropy.
- **Optimizer:** Adam optimizer.
- **Evaluation metric:** Accuracy.

## Model Training and Performance

### Training History
| Epoch | Accuracy | Loss | Validation Accuracy |
|-------|---------|------|---------------------|
| 1 | 0.8935 | 1.1520 | 0.9505 |
| 2 | 0.9671 | 0.1210 | 0.9617 |
| 3 | 0.9802 | 0.0835 | 0.9598 |
| 4 | 0.9826 | 0.0782 | 0.9574 |
| 5 | 0.9869 | 0.0589 | 0.9607 |

### Final Model Evaluation
- **Test Accuracy:** 95.98%
- **Validation Accuracy:** 96.06%
