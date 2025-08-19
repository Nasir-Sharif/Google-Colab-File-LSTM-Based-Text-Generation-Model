# LSTM-based Text Generation Model

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Prepare the Dataset](#prepare-the-dataset)
  - [Run the Script](#run-the-script)
  - [Generate Text](#generate-text)
  - [Evaluate the Model](#evaluate-the-model)
- [Project Structure](#project-structure)
- [Code Explanation](#code-explanation)
  - [Data Preprocessing](#data-preprocessing)
  - [LSTM Model](#lstm-model)
  - [Training](#training)
  - [Prediction](#prediction)
  - [Evaluation](#evaluation)
- [Example Output](#example-output)
  - [Training Output](#training-output)
  - [Prediction Output](#prediction-output)
  - [Accuracy Output](#accuracy-output)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [Resources and Links](#resources-and-links)
- [Contact](#contact)
- [License](#license)

## Project Overview

This project implements a Long Short-Term Memory (LSTM) neural network for text generation using [PyTorch](https://pytorch.org/) and [NLTK](https://www.nltk.org/). The model is trained on a dataset containing information about the **Data Science Mentorship Program (DSMP 2023)**. It processes input text, tokenizes it, and generates a sequence of words by predicting the next word based on the given context.

The code includes data preprocessing, model training, prediction, and evaluation of the model's accuracy. The primary goal is to demonstrate how an LSTM model can be used for natural language processing tasks, specifically next-word prediction.

## Features

- **Text Preprocessing**:
  - Tokenization using NLTK's `word_tokenize`.
  - Vocabulary creation with an `<unk>` token for unknown words.
  - Conversion of text to numerical indices.
  - Pre-padding sequences to ensure uniform length.
- **Custom Dataset**:
  - A PyTorch `Dataset` class to handle input and output sequences.
  - DataLoader for efficient batch processing.
- **LSTM Model**:
  - Custom LSTM architecture with an embedding layer, LSTM layer, and a fully connected layer.
  - Configurable hyperparameters (embedding size, hidden size, etc.).
- **Training**:
  - Training loop with Adam optimizer and CrossEntropyLoss.
  - Support for GPU acceleration if available.
- **Prediction**:
  - Generates the next word given an input text sequence.
  - Supports iterative text generation for multiple tokens.
- **Evaluation**:
  - Computes model accuracy on the training dataset.

## Prerequisites

Before running the code, ensure you have the following installed:

- [Python](https://www.python.org/) 3.7 or higher
- [PyTorch](https://pytorch.org/) (`pip install torch`)
- [NLTK](https://www.nltk.org/) (`pip install nltk`)
- [NumPy](https://numpy.org/) (`pip install numpy`)

Install the required dependencies using:

```bash
pip install torch nltk numpy
```

Download the required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/lstm-text-generation.git
   cd lstm-text-generation
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:
   - The code uses a sample text document about the DSMP 2023 program. You can replace it with your own text data by updating the `document` variable in the script.

## Usage

### Prepare the Dataset

The dataset is a text document containing details about the DSMP 2023 program. You can modify the `document` variable in the script to use your own text data. The text is tokenized, converted to numerical indices, and padded to a fixed length (maximum sequence length: 62).

### Run the Script

Execute the main script to preprocess the data, train the model, and generate predictions:

```bash
python lstm_text_generation.py
```

### Generate Text

After training, the model can generate text by predicting the next word. Example:

```python
input_text = "The course follows a monthly"
output = prediction(model, vocab, input_text)
print(output)
```

To generate multiple tokens iteratively:

```python
input_text = "hi how are"
for i in range(10):
    output_text = prediction(model, vocab, input_text)
    print(output_text)
    input_text = output_text
```

### Evaluate the Model

The script includes a function to calculate the model's accuracy on the training dataset:

```python
accuracy = calculate_accuracy(model, dataloader, device)
print(f"Model Accuracy: {accuracy:.2f}%")
```

## Project Structure

```
lstm-text-generation/
├── lstm_text_generation.py  # Main script containing the LSTM model and logic
├── README.md                # Project documentation
├── requirements.txt         # List of dependencies
```

## Code Explanation

### Data Preprocessing

- **Tokenization**: The input text is tokenized into words using NLTK's `word_tokenize`.
- **Vocabulary Creation**: A vocabulary dictionary maps each unique word to an index, with `<unk>` for unknown words.
- **Numerical Representation**: Sentences are converted to numerical indices using the vocabulary.
- **Padding**: Sequences are pre-padded with zeros to match the maximum sequence length (62 in the provided dataset).
- **Dataset and DataLoader**: A custom `Dataset` class and PyTorch `DataLoader` handle input-output pairs for training.

### LSTM Model

- **Architecture**:
  - **Embedding Layer**: Converts word indices to dense vectors (embedding size: 100).
  - **LSTM Layer**: Processes the sequence with a hidden size of 150.
  - **Fully Connected Layer**: Maps the LSTM output to the vocabulary size for next-word prediction.
- **Forward Pass**: The model processes the input sequence, generates hidden states, and outputs logits for each word in the vocabulary.

### Training

- **Hyperparameters**:
  - Epochs: 50
  - Learning Rate: 0.001
  - Batch Size: 32
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Device**: Automatically uses GPU (CUDA) if available, otherwise CPU.

### Prediction

- The `prediction` function tokenizes input text, converts it to a padded numerical sequence, and predicts the next word using the trained model.
- Iterative text generation is supported by feeding the predicted output back as input.

### Evaluation

- The `calculate_accuracy` function computes the percentage of correctly predicted words in the training dataset.

## Example Output

### Training Output

```
Epoch: 1, Loss: 1234.5678
Epoch: 2, Loss: 1123.4567
...
Epoch: 50, Loss: 456.7890
```

### Prediction Output

```python
input_text = "The course follows a monthly"
# Example output: "The course follows a monthly subscription"
```

### Accuracy Output

```python
Model Accuracy: 85.23%
```

## Limitations

- **Dataset Size**: The model is trained on a small dataset, which may limit its generalization to unseen text.
- **Overfitting**: With a small dataset and many epochs, the model may overfit. Consider adding regularization or early stopping.
- **Single-Word Prediction**: The model predicts one word at a time, which may not capture long-term dependencies effectively.
- **No Fine-Tuning**: The hyperparameters are fixed and may not be optimal for all datasets.

## Future Improvements

- Add support for larger datasets to improve model robustness.
- Implement early stopping to prevent overfitting.
- Experiment with different architectures (e.g., bidirectional LSTM or [Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)).
- Add hyperparameter tuning (e.g., grid search for learning rate, embedding size).
- Support for generating longer sequences or full sentences.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and UPDATE: Updated README.md file with improved formatting, headings, subheadings, and links.