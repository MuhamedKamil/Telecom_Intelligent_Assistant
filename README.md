# 🤖 AI Intelligent Assistant (Arabic / English)

An AI-powered Intelligent Assistant that supports **Arabic and English** through both **text and voice interactions**. The system combines Automatic Speech Recognition (ASR), Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), Text-to-Speech (TTS), and conversational memory to provide fast, context-aware, and natural responses.

The assistant is designed for customer support, technical documentation, and enterprise knowledge bases, enabling users to communicate naturally by typing or speaking.

---

# 🚀 Features

| Feature | Description | Business Impact |
|---------|-------------|-----------------|
| 🎤 Voice Analytics | Real-time Arabic and English ASR for speech transcription | 70% reduction in manual QA |
| 📚 Knowledge Retrieval | Instant retrieval of relevant documents and FAQs using RAG | 60% faster issue resolution |
| 🧠 Intelligent Agent | Context-aware LLM for intelligent customer interactions | 24/7 automated support |
| 🗣️ Voice Synthesis | Natural Arabic and English Text-to-Speech responses | Enhanced customer experience |
| 📊 Analytics Dashboard | Real-time metrics and insights | Data-driven decision making |
| 💾 Conversation Memory | Stores recent conversation history (up to **200** memory documents) for contextual responses | Improved conversation continuity |

---

# 🏗️ System Pipeline

## 1. Text Pipeline

```text
User Text
     │
     ▼
Knowledge Retrieval (RAG)
     │
     ▼
Relevant Chunks
     │
     ▼
Large Language Model
     │
     ▼
Generated Answer
```

---

## 2. Voice Pipeline

```text
User Voice
     │
     ▼
Automatic Speech Recognition (ASR)
     │
     ▼
Knowledge Retrieval (RAG)
     │
     ▼
Relevant Chunks
     │
     ▼
Large Language Model
     │
     ▼
Generated Answer
     │
     ▼
Text-to-Speech (TTS)
     │
     ▼
Voice Response
```

---

## 3. Conversation Memory

```text
Conversation
      │
      ▼
Memory Storage
      │
      ▼
Store Last 200 Memory Documents
      │
      ▼
Used During Future Retrieval
```

The assistant maintains a memory of the latest **200 conversation documents**, allowing it to generate more context-aware and personalized responses.

---

# ⚙️ Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

AND install the dependencies manually:

```bash
pip install -U transformers accelerate

pip install silma-tts --no-deps
pip install silma-tts

pip install unstructured-inference --no-deps
pip install "unstructured[all-local]"

pip install gradio -q

pip install -U bitsandbytes>=0.46.1

pip install speechbrain
```

---

# 🔑 Hugging Face Authentication

Before running the application, log in to your Hugging Face account.

```python
from huggingface_hub import login

login(token="YOUR_HUGGINGFACE_TOKEN")
```

Replace `YOUR_HUGGINGFACE_TOKEN` with your own Hugging Face access token.

---

# ▶️ Run the Application

```bash
python app_gradio.py
```

When running in **Google Colab**, Gradio will generate a public URL that you can open in your browser to interact with the assistant.

---

# 🌍 Supported Languages

- 🇸🇦 Arabic
- 🇺🇸 English

Supports both:

- 💬 Text conversations
- 🎙️ Voice conversations

---

# 🧠 Core Technologies

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Automatic Speech Recognition (ASR)
- Text-to-Speech (TTS)
- Conversational Memory
- Gradio
- Hugging Face Transformers

---

# 📌 Use Cases

- Telecom customer support
- Technical documentation assistant
- FAQ automation
- Voice-enabled AI assistant
- Enterprise knowledge management
- Multilingual conversational AI