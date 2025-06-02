# vita
AI Call Agent for Outbound Call: For iBuild Hackathon


## Project Description: AI Inbound and Outbound Call Agent

The "AI Inbound and Outbound Call Agent" is an advanced conversational AI system designed to autonomously handle both incoming and outgoing voice calls, leveraging state-of-the-art speech and language technologies for natural, context-aware customer interactions. The system is implemented as a modular Python application, orchestrated within a Colab environment, and integrates several industry-leading AI components for end-to-end voice automation.

**Key Technical Features:**

- **Speech Recognition:** Utilizes OpenAI Whisper for robust, multilingual voice-to-text transcription, enabling accurate capture of user utterances from real-time audio streams.
- **Speech Synthesis:** Employs Google Text-to-Speech (gTTS) and optionally advanced neural TTS engines (such as ElevenLabs, Bark AI, and edge-tts) to generate natural-sounding, contextually appropriate audio responses.
- **Intent Detection:** Implements zero-shot classification using large language models (LLMs, e.g., Facebook BART via Huggingface Transformers) to categorize user queries into predefined intent buckets (e.g., Greeting, Symptom Profiling, Objection Handling, Cost Inquiry, etc.).
- **Semantic Matching:** Applies keyword and cosine similarity matching (using scikit-learn's CountVectorizer and cosine_similarity) to map user queries to the most relevant predefined questions and retrieve corresponding answers, enhancing response accuracy and relevance.
- **Conversational Memory:** Supports call memorization by dynamically concatenating new keywords and context from ongoing dialogues, enabling context-aware, multi-turn conversations and improved response continuity.
- **Multilingual Support:** Integrates language detection (langDetect), translation (Huggingface MarianMT), and multilingual TTS/ASR pipelines, allowing seamless handling of calls in multiple languages.
- **Emotion Recognition:** Incorporates open-source emotion detection models (e.g., Wav2Vec2 + EmoNet for speech, GoEmotions + DistilBERT for text) to analyze and adapt to the emotional tone of the caller, personalizing the interaction.
- **Personalization:** Connects to external data sources (e.g., Google Sheets via gspread) to retrieve caller-specific information (such as first names), enabling personalized greetings and context-aware responses.
- **Interactive Voice Workflow:** Features a browser-based audio recording and playback system using JavaScript and IPython display hooks, enabling real-time voice input/output within the notebook environment.

**System Workflow Overview:**

1. **Call Initiation:** The agent either receives or initiates a call, prompting the user for input.
2. **Voice Capture & Transcription:** User speech is recorded, transcribed to text using Whisper, and preprocessed for downstream tasks.
3. **Intent Analysis:** The transcribed text is classified into one of several intent categories using zero-shot LLM classification.
4. **Semantic Matching & Response Generation:** The agent matches the input to predefined Q&A pairs using semantic similarity and retrieves the most relevant answer.
5. **Speech Synthesis & Playback:** The selected response is synthesized into speech and played back to the user in real time.
6. **Context Management:** The system maintains conversational context and memory, supporting multi-turn, coherent dialogues.
7. **Personalization & Emotion Adaptation:** Responses are tailored based on caller data and detected emotional state, enhancing user engagement.

**Extensibility:**

The architecture is designed for extensibility, allowing integration of more advanced voice synthesis engines, additional languages, and richer emotional intelligence modules. The system can be adapted for various domains, including healthcare, customer support, and sales automation, by updating the intent buckets and Q&A knowledge base.

**Core Technology Stack:**

- Python, Jupyter/Colab
- OpenAI Whisper (ASR)
- Huggingface Transformers (LLMs, MarianMT)
- gTTS, edge-tts, ElevenLabs/Bark AI (TTS)
- spaCy (NLP preprocessing)
- scikit-learn (semantic similarity)
- gspread (Google Sheets integration)
- JavaScript (browser-based audio I/O)
- Wav2Vec2, EmoNet, GoEmotions (emotion recognition)

This project demonstrates a comprehensive, production-ready framework for AI-powered voice agents, combining speech, language, and emotion AI to deliver human-like, automated call experiences.
