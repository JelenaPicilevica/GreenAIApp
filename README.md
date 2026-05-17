# GreenAIApp

Cache-aware Green AI framework for reducing redundant Large Language Model (LLM) inference through 
semantic caching, semantic drift detection, and adaptive prompt/response reuse.

This project was developed as part of a Master's thesis focused on improving LLM inference efficiency 
while reducing token usage, estimated energy consumption, and CO₂ emissions.

---

# Features

- Semantic caching using sentence embeddings
- Prompt and response reuse
- Semantic drift detection
- Lightweight neural network classifier
- Transformer-based fallback model
- Hybrid decision mechanism
- Streamlit web interface
- Token usage and efficiency evaluation
- Persistent cache storage

---

# Technologies Used

## Programming Language
- Python

## Machine Learning & NLP
- Scikit-learn
- Sentence Transformers
- Transformers (Hugging Face)
- NumPy
- Pandas

## Visualization & Analysis
- Matplotlib
- Seaborn

## Application Interface
- Streamlit

## APIs & Embeddings
- OpenAI Embeddings API
- text-embedding-3-small

## Development Tools
- PyCharm
- Git
- GitHub
- Google Colab

---

# Project Workflow

PAWS dataset -> Train transformer-based semantic drift model -> Train lightweight neural network -> Combine models into hybrid fallback mechanism
-> Generate experimental semantic cache dataset from SQuAD prompts -> Run Streamlit application

---

# Datasets Used

## PAWS Dataset
Used for:
- semantic drift detection
- neural network training
- transformer training

Dataset:
https://huggingface.co/datasets/google-research-datasets/paws

---

## SQuAD Dataset
Used for:
- experimental semantic cache generation
- prompt and response storage

Dataset:
https://huggingface.co/datasets/rajpurkar/squad

---

## LMSYS Chat-1M Dataset
Used for:
- token statistics
- analytical efficiency evaluation

Dataset:
https://huggingface.co/datasets/lmsys/lmsys-chat-1m

---

# Installation

Clone repository:

```bash
git clone https://github.com/JelenaPicilevica/GreenAIApp.git
cd GreenAIApp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create `.env` file:

```text
OPENAI_API_KEY=your_api_key
```

---

# Training and Execution Order

## 1. Train Transformer Model

Train transformer-based semantic drift detector first:

```bash
python train_transformer.py
```

Training metrics and evaluation results will be stored in logs.

---

## 2. Train Neural Network

After transformer training, train lightweight neural network with transformer fallback support:

```bash
python train_neural_network.py
```

The hybrid model uses:
- neural network as primary model
- transformer model as fallback for uncertain predictions

Training and evaluation results will be available in logs.

---

## 3. Generate Semantic Cache

Generate semantic cache dataset from SQuAD prompts:

```bash
python build_cache.py
```

Generated cache contains:
- prompts
- responses
- embeddings
- token statistics

Example output:

```text
data/cache_dataset.csv
```

---

# Running the Application

Start Streamlit application:

```bash
streamlit run app.py
```

---

# Framework Architecture

The framework consists of:
- sentence embedding generation
- semantic similarity retrieval
- semantic drift detection
- semantic caching
- transformer fallback mechanism
- adaptive reuse strategy
- user-controlled reuse decisions
- LLM interaction module

The system supports:
- direct cached answer reuse
- optimized prompt reuse
- standard LLM inference when reuse is not suitable

---

# Experimental Goals

The framework evaluates:
- token reduction
- computational efficiency
- estimated energy savings
- estimated CO₂ reduction
- semantic drift detection accuracy
- hybrid model performance

---

# Notes

- The semantic cache is persisted locally as a dataset containing prompt-response pairs, embeddings, and token statistics.
- Cached prompt-response pairs and embeddings are stored persistently for reuse between sessions and experiments.
- Energy and CO₂ values are analytical estimations based on token reduction.
- The project is intended as a research prototype and controlled experimental framework.

---

# Author

Jeļena Picilēviča
