# Agentic AI Research Support Tool: Singlish Student Assistant & LLM Evaluator

**Module:** IT41043 - Intelligent Systems (Agentic AI)  
**Institution:** Horizon Campus Sri Lanka  
**Student Name:** Kawshalya LiyanaArachchi (ITBIN-2313-0056)  
**Partner Name:** MNM Dias  
**Live Application URL:** [Streamlit Community Cloud Link Placeholder]  

---

## 1. Project Overview & Research Context

This application serves as an Agentic AI system and research support tool for evaluating Large Language Models (LLMs) processing code-mixed Sinhala-English ("Singlish") queries within Sri Lankan higher education student support services.

It addresses key administrative challenges by offering:
1. **Singlish Academic Support Assistant:** An intelligent agentic pipeline answering student queries regarding prerequisites, repeat exams, medical submissions, and welfare guidelines grounded in official handbook documentation via Retrieval-Augmented Generation (RAG).
2. **NLP Research Benchmarking Tool:** An evaluation suite analyzing tokenization breakdowns, structural parsing errors, and intent confidence when commercial and open-source LLMs encounter localized code-mixed dialects.

---

## 2. Agentic Design Patterns Implemented

This project implements three distinct agentic design patterns across four specialized agents:

1. **Router Pattern (`src/agents.py` - Agent 1):** Uses an ultra-low-latency model (`llama-3.1-8b-instant` on Groq) to classify incoming user queries into specific intents and decide if vector store retrieval is necessary.
2. **ReAct / Tool-Use Pattern (`src/agents.py` - Agent 2):** Queries the local Chroma vector store (`src/rag_pipeline.py`) to retrieve ground-truth university handbook contexts when requested by the router.
3. **Reflection / Self-Critique Pattern (`src/agents.py` - Agent 3):** Uses a high-reasoning model (`claude-3.5-sonnet` on OpenRouter) to inspect the query for Singlish code-mixed terms, critique potential tokenization breakdowns, and calculate an intent confidence score.

---

## 3. Agent-to-Agent Communication Sequence

The system uses structured JSON and Pydantic schemas to exchange payloads between agents.

```mermaid
sequenceDiagram
    autonumber
    actor Student as Undergraduate / User
    participant Router as Agent 1: Intent Router (Groq Llama 3.1)
    participant Retriever as Agent 2: RAG Retriever (Chroma DB Tool)
    participant Critique as Agent 3: Singlish Reflection Agent (OpenRouter Claude 3.5)
    participant Synthesizer as Agent 4: Final Synthesizer (OpenRouter Claude 3.5)

    Student->>Router: Submit Singlish Query
    Router->>Router: Classify Intent & Determine RAG Need
    alt RAG Required
        Router->>Retriever: Pass Normalized Query Payload
        Retriever->>Retriever: Similarity Search over Chroma Vector DB
        Retriever-->>Synthesizer: Return Relevant Handbook Chunks
    end
    Router->>Critique: Pass Original Query
    Critique->>Critique: Analyze Tokenization & Extract Singlish Terms
    Critique-->>Synthesizer: Return Structural Critique Payload
    Synthesizer->>Synthesizer: Synthesize Final Grounded Answer
    Synthesizer-->>Student: Display Structured UI Response
