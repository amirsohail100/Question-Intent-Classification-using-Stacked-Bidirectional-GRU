# Question Intent Classification using Stacked Bidirectional GRU

An advanced Natural Language Processing (NLP) deep learning project designed to process input questions and classify them into their respective target categories using Gated Recurrent Units (GRU).

---

## 🌟 Key Highlights

- **Model Architecture**: Stacked Bidirectional GRU (`Bidirectional(GRU(128))` + `Bidirectional(GRU(64))`) with Dropout regularization.
- **Fast & Efficient**: GRUs offer faster convergence and lower computational complexity compared to traditional LSTMs while capturing long-term sequential dependencies.
- **Robust Feature Extraction**: Dual-stage Dense feed-forward layers (`32` and `16` units with `ReLU`) before output prediction.

---

## 🏗️ Neural Network Architecture

The deep learning model is built using the Keras `Sequential` API:

1. **Embedding Layer**: Transforms tokenized text sequences into dense vector representations (`output_dim=Embedding_dim`).
2. **First Bidirectional GRU Layer (`128 units`, `return_sequences=True`)**: Captures contextual sequential features in both forward and backward directions across the input text.
3. **Dropout Layer (`0.3`)**: Prevents overfitting by randomly dropping 30% of node connections during training.
4. **Second Bidirectional GRU Layer (`64 units`, `return_sequences=False`)**: Synthesizes the sequence outputs into a consolidated feature vector.
5. **Dropout Layer (`0.2`)**: Provides additional regularization.
6. **Dense Layers (`32` and `16` units with `ReLU`)**: Extracts high-level non-linear representations.
7. **Output Layer (`1 unit`)**: Predicts the target class output.

```python
model = Sequential([
    Embedding(input_dim=Max_words, output_dim=Embedding_dim, input_length=Max_len),
    Bidirectional(GRU(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(GRU(64, return_sequences=False)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])
```

````
## 📦 Required Pipeline Files

To run this application properly, ensure the following pipeline files are present in your project directory:

- `model.h5` (Trained ANN Model)
- `tokenizer.pkl` (Tokenizer file)
- `columns.pkl` (Data preprocessing layout)

## 🚀 Getting Started

1. Clone this repository.
2. Place the required pipeline files (`model.h5`, `tokenizer.pkl`, `columns.pkl`) in the root directory.
3. Run the application script to start the local server.

## Cloud Deployment

```bash
git clone https://github.com/amirsohail100/Question-Intent-Classification-using-Stacked-Bidirectional-GRU.git
````

```bash
cd Question-Intent-Classification-using-Stacked-Bidirectional-GRU
```

```bash
python install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License.

## 📝 Author

👤 **Amir Sohail**

Question Intent Classifier using Bidirectional GRU 🚀 Built an NLP model to classify user questions into intent/category classes. Powered by a Stacked Bidirectional GRU architecture (128 & 64 units) with Dropout layers (0.3/0.2) to capture sequence context in both directions while preventing overfitting. Built with TensorFlow/Keras. 🎯
