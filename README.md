# Agentic AI Research Support Tool: Singlish Student Assistant & LLM Evaluator

**Module:** IT41043 - Intelligent Systems (Agentic AI)  
**Institution:** - Horizon Campus  
**Student Name:** - ITBIN-2313-0056 - Kawshalya LiyanaArachchi   
                    ITBIN-2313-0023 - Nipuni Malsha Dias  
**Live Application URL:** https://singlish-student-support-agent-nsy7edd8wn5htojuge5cms.streamlit.app/

---

## 📌 1. Executive Summary & Research Context

Sri Lankan university students frequently communicate using **Code-Mixed Singlish** (Romanized Sinhala combined with English, e.g., *"Repeat exam register karanne kohomada form A2 ganna codehada?"*). Standard Large Language Model (LLM) pipelines often struggle with tokenization breakdown, morphological parsing, and intent misalignment when handling such informal code-mixed text.

This project implements a **Multi-Agent Retrieval-Augmented Generation (RAG) Architecture** designed to evaluate, benchmark, and resolve code-mixed Singlish queries for Sri Lankan higher education support services. The system comprehends Singlish queries, retrieves verified policy context from university handbooks (PDFs), performs linguistic critique, and outputs clear, professional English responses.

---

## 🏗️ 2. Multi-Agent Design Patterns & System Architecture

The architecture consists of four specialized AI agents operating in an orchestrated pipeline using **Groq (Llama 3.1 & 3.3)** and **OpenRouter**:

| Pattern Used | Function / Code Location | Description |
| :--- | :--- | :--- |
| **1. Router Pattern** | `src/agents.py` → `agent_1_intent_router()` | Classifies input query into `academic_query`, `research_benchmark`, or `general`. Determines if RAG context lookup is required. |
| **2. Tool-Use Pattern** | `src/agents.py` → `agent_2_rag_retriever()` | Leverages an external vector tool (`ChromaDB`) to search indexed university handbook PDFs ($k=2$ chunks). |
| **3. Reflection / Critique Pattern** | `src/agents.py` → `agent_3_reflection_critique()` | Analyzes Singlish tokens, evaluates morphological breakdown points, and calculates confidence scores. |
| **4. Orchestrator-Worker Pattern** | `src/agents.py` → `run_multi_agent_pipeline()` | Manages sequential agent-to-agent payload forwarding and aggregates structured outputs for final synthesis. |

---

## 🔄 3. Agent-to-Agent Communication & Sequence Flow

The system employs structured payload exchanges between agents (A2A/MCP-inspired schema exchange via Pydantic):

```text
[Student User]
      │ (Code-Mixed Singlish Query)
      ▼
┌───────────────────────────────────────┐
│ Agent 1: Intent Router                │
│ Code: agent_1_intent_router()         │
│ Payload: RouterOutput (JSON Schema)   │
└───────────────────┬───────────────────┘
                    │
      ┌─────────────┴─────────────┐
      ▼                           ▼
(requires_rag = True)       (requires_rag = False)
      │                           │
┌─────┴──────────────────┐        │
│ Agent 2: RAG Retriever │        │
│ Code: agent_2_rag()    │        │
│ Payload: List[Context] │        │
└─────┬──────────────────┘        │
      │                           │
      └─────────────┬─────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ Agent 3: Reflection Agent             │
│ Code: agent_3_reflection_critique()   │
│ Payload: CritiqueOutput (JSON Schema) │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ Agent 4: Final Synthesizer            │
│ Code: agent_4_final_synthesizer()     │
│ Combines: Upstream Pydantic Payloads  │
└───────────────────┬───────────────────┘
                    │
                    ▼
[Streamlit Frontend Output]
```

---

## 📊 4. LLM Model Selection Strategy & Comparison

| Sub-task | Model (Provider) | Latency | Cost / Token | Context Window | Reasoning Quality & Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intent Routing & Classification (Agent 1)** | `llama-3.1-8b-instant` (Groq) | ~0.2s | Free / Low | 128k tokens | High accuracy for structured JSON intent parsing with minimal response latency. |
| **Linguistic Reflection & Critique (Agent 3)** | `llama-3.3-70b-versatile` (Groq / OpenRouter) | ~0.8s | Free / Low | 128k tokens | Superior morphological reasoning needed to identify code-mixed Singlish tokenization breakdown. |
| **Final Answer Synthesis (Agent 4)** | `llama-3.3-70b-versatile` (Groq / OpenRouter) | ~1.0s | Free / Low | 128k tokens | High reasoning quality required to combine student query, RAG context, and critique into a clear answer. |

---

## 📚 5. RAG Integration & Evaluation

* **Corpus Scope:** 20+ University Handbook PDFs & Academic Regulation documents stored in `data/`.
* **Vector Store:** ChromaDB (`chromadb/`).
* **Embedding Model:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`.
* **Chunking Strategy:** `RecursiveCharacterTextSplitter` (Chunk Size: 600 characters, Overlap: 100 characters).

### 🧪 Retrieval Evaluation (5 Sample Queries)

| # | Sample Query (Code-Mixed / English) | Context Retrieved | Retrieved Context Relevant? | Comments / Observations |
| :-: | :--- | :--- | :-: | :--- |
| 1 | *Repeat exam register karanne kohomada form A2 ganna codehada?* | Form A2 registration deadline & fee policy chunk. | **Yes** | Successfully mapped Singlish intent to official repeat exam procedures. |
| 2 | *IT41043 prerequisite subjects monawada?* | IT41043 syllabus & prerequisite requirement chunk. | **Yes** | Extracted exact required prerequisite modules (IT3010, IT3020). |
| 3 | *GPA eka calculate karana formula eka mokakda?* | Academic Regulations section on Grade Point Average calculation. | **Yes** | Fetched weighted credit point calculation rules accurately. |
| 4 | *Singlish LLM tokenization breakdown wenne ai?* | Research paper excerpt on Romanized Sinhala morphology. | **Yes** | Successfully retrieved academic research background context. |
| 5 | *Campus office open karanne keeyatada?* | General university operating hours & contact chunk. | **Yes** | Retrieved campus administrative operating schedule accurately. |

---

## ⚙️ 6. Local Setup & Ingestion Instructions

### 1. Repository Setup
```bash
git clone [https://github.com/Liyanaarachchi04/singlish-student-support-agent.git](https://github.com/Liyanaarachchi04/singlish-student-support-agent.git)
cd singlish-student-support-agent
python -m venv venv

# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Secrets
Create `.streamlit/secrets.toml` in your root directory:
```toml
GROQ_API_KEY = "gsk_your_groq_api_key"
OPENROUTER_API_KEY = "sk-or-v1_your_openrouter_api_key"
```

### 4. Ingest University PDF Handbooks
Drop all university handbooks and academic rule PDF files into the `data/` directory, then run the ingestion pipeline:
```bash
python -m src.rag_pipeline
```

### 5. Run Streamlit App
```bash
python -m streamlit run app.py
```

---

## ⚠️ 7. Known Limitations & Future Work

1. **Orthographic Variability:** Code-mixed Singlish lacks standardized spelling (e.g., *"register karanne"* vs. *"register koranne"*), which can occasionally affect embedding similarity matching in ChromaDB.
2. **Multi-Column PDF Formatting:** Heavy PDF tables with complex multi-column layouts may require table-aware parsers for optimal chunk extraction.
3. **External API Latency:** Cascading four multi-agent prompts sequentially relies on cloud LLM availability and network latency.

---

## 📁 8. Repository Structure

```text
singlish-student-support-agent/
├── .streamlit/
│   └── secrets.toml         # Environment API Keys (Git ignored)
├── data/                    # PDF Handbooks & Text Documents directory
├── chromadb/                # Vector Database Store (Embedded chunks)
├── src/
│   ├── __init__.py
│   ├── agents.py            # Multi-Agent Pipeline & LangChain Logic
│   └── rag_pipeline.py      # PDF Loader, Text Splitter & ChromaDB Ingestion
├── app.py                   # Streamlit UI Frontend
├── requirements.txt         # Dependencies list for Streamlit Cloud
├── .gitignore
└── README.md                # Project System Documentation
```

---
*Developed for IT41043 Research & Assignment Evaluation.*
