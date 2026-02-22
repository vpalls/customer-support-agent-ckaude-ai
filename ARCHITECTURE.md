# 🏗️ ARCHITECTURE — Customer Support Router Agentic RAG System

> Complete technical architecture documentation for the intelligent customer support agent
> powered by Claude AI, LangGraph, ChromaDB, and Chainlit.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Component Deep Dive](#3-component-deep-dive)
4. [LangGraph Agent Workflow](#4-langgraph-agent-workflow)
5. [Data Flow & Sequence Diagram](#5-data-flow--sequence-diagram)
6. [RAG Pipeline Details](#6-rag-pipeline-details)
7. [Technology Stack](#7-technology-stack)
8. [File Structure & Module Map](#8-file-structure--module-map)
9. [Deployment Architecture](#9-deployment-architecture)
10. [Security & Configuration](#10-security--configuration)
11. [Scalability Considerations](#11-scalability-considerations)

---

## 1. System Overview

This system is an **Agentic RAG (Retrieval-Augmented Generation)** application that handles
customer support queries through an intelligent multi-step pipeline. It combines:

- **Query Classification** — Categorizes incoming questions into domains
- **Sentiment Analysis** — Detects user emotional tone for escalation routing
- **Conditional Routing** — Dynamically selects the appropriate response handler
- **RAG Retrieval** — Fetches relevant knowledge from a vector store
- **LLM Generation** — Produces context-aware responses using Claude

The system operates as a **stateful agent graph** where each node performs a discrete function,
and edges define the routing logic between nodes.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                  │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │                    Chainlit Web UI                            │   │
│   │   ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐  │   │
│   │   │ Welcome  │  │  Chat    │  │  Steps    │  │  Badges   │  │   │
│   │   │ Screen   │  │  Window  │  │  Display  │  │  (meta)   │  │   │
│   │   └─────────┘  └──────────┘  └───────────┘  └───────────┘  │   │
│   └──────────────────────────┬───────────────────────────────────┘   │
│                              │ WebSocket + HTTP                      │
└──────────────────────────────┼───────────────────────────────────────┘
                               │
┌──────────────────────────────┼───────────────────────────────────────┐
│                     APPLICATION LAYER                                 │
│                              │                                       │
│   ┌──────────────────────────▼───────────────────────────────────┐   │
│   │                  app.py (Chainlit Server)                     │   │
│   │                                                              │   │
│   │   on_chat_start() ──→ Session init + Welcome message         │   │
│   │   on_message()    ──→ Run agent pipeline + Format response   │   │
│   │   on_chat_end()   ──→ Session cleanup                        │   │
│   └──────────────────────────┬───────────────────────────────────┘   │
│                              │                                       │
│   ┌──────────────────────────▼───────────────────────────────────┐   │
│   │                  agent.py (LangGraph Agent)                   │   │
│   │                                                              │   │
│   │   ┌────────────┐    ┌──────────┐    ┌────────────────────┐  │   │
│   │   │ Categorize │───▶│ Analyze  │───▶│   Smart Router     │  │   │
│   │   │  Inquiry   │    │Sentiment │    │ (Conditional Edge) │  │   │
│   │   └────────────┘    └──────────┘    └──┬──┬──┬──┬────────┘  │   │
│   │                                        │  │  │  │           │   │
│   │          ┌─────────────────────────────┘  │  │  │           │   │
│   │          ▼           ▼           ▼        ▼                 │   │
│   │   ┌──────────┐┌──────────┐┌──────────┐┌──────────┐         │   │
│   │   │Technical ││ Billing  ││ General  ││ Escalate │         │   │
│   │   │ Response ││ Response ││ Response ││ to Human │         │   │
│   │   └────┬─────┘└────┬─────┘└────┬─────┘└──────────┘         │   │
│   │        │           │           │                            │   │
│   │        └───────────┴───────────┘                            │   │
│   │                    │ RAG retrieval                           │   │
│   └────────────────────┼────────────────────────────────────────┘   │
│                        │                                             │
└────────────────────────┼─────────────────────────────────────────────┘
                         │
┌────────────────────────┼─────────────────────────────────────────────┐
│                   DATA & AI LAYER                                     │
│                        │                                             │
│   ┌────────────────────▼─────────────────────────────────────────┐   │
│   │              knowledge_base_loader.py                         │   │
│   │                                                              │   │
│   │   ┌───────────────┐     ┌──────────────────────────────┐    │   │
│   │   │  JSON Loader  │────▶│  HuggingFace Embeddings      │    │   │
│   │   │ (Knowledge    │     │  (all-MiniLM-L6-v2)          │    │   │
│   │   │   Base)       │     │  Local, CPU-based, Free      │    │   │
│   │   └───────────────┘     └──────────┬───────────────────┘    │   │
│   │                                    │                        │   │
│   │                         ┌──────────▼───────────────────┐    │   │
│   │                         │       ChromaDB               │    │   │
│   │                         │   (Persistent Vector Store)  │    │   │
│   │                         │                              │    │   │
│   │                         │  Collection: knowledge_base  │    │   │
│   │                         │  Distance:   cosine          │    │   │
│   │                         │  Storage:    Docker volume    │    │   │
│   │                         └──────────────────────────────┘    │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │              Claude API (Anthropic)                           │   │
│   │                                                              │   │
│   │   Model:     claude-sonnet-4-20250514                        │   │
│   │   Features:  Structured Output, Chat Completion              │   │
│   │   Calls:     2 per query (classify + generate)               │   │
│   │              + 1 for sentiment analysis                      │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Deep Dive

### 3.1 Chainlit UI Layer (`app.py`)

The presentation layer manages user sessions, renders the chat interface, and orchestrates
communication between the user and the LangGraph agent.

```
Responsibilities:
├── Session Management
│   ├── on_chat_start()   → Initialize session, display welcome message
│   ├── on_message()      → Route user input to agent, format output
│   ├── on_chat_end()     → Cleanup session state
│   └── on_stop()         → Handle interrupted responses
│
├── Rich UI Elements
│   ├── cl.Step()         → Show processing pipeline steps (Categorize, Analyze)
│   ├── cl.Text()         → Inline markdown tables (capabilities card)
│   ├── cl.Message()      → Chat messages with metadata badges
│   └── Markdown badges   → Category emoji + Sentiment emoji per response
│
└── Async/Thread Management
    └── asyncio.to_thread() → Run sync LangGraph agent without blocking event loop
```

**Key Design Decision:** The LangGraph agent runs synchronously (it streams events), so we
wrap it in `asyncio.to_thread()` to prevent blocking Chainlit's async event loop.

### 3.2 LangGraph Agent (`agent.py`)

The core intelligence layer implements a state machine using LangGraph's `StateGraph`.

```
Agent State (CustomerSupportState):
├── customer_query    : str   →  Raw user input
├── query_category    : str   →  "Technical" | "Billing" | "General"
├── query_sentiment   : str   →  "Positive" | "Neutral" | "Negative"
└── final_response    : str   →  Generated response text

Pydantic Schemas (for structured LLM output):
├── QueryCategory     →  categorized_topic: Literal['Technical','Billing','General']
└── QuerySentiment    →  sentiment: Literal['Positive','Neutral','Negative']
```

**Claude Integration:** Uses `langchain-anthropic`'s `ChatAnthropic` with `.with_structured_output()`
for reliable JSON-mode classification. This replaces the original `ChatOpenAI` + `with_structured_output`
pattern and works identically with Claude's tool-use based structured output.

### 3.3 Knowledge Base & Vector Store (`knowledge_base_loader.py`)

The data layer handles document ingestion, embedding, and retrieval.

```
Pipeline:
JSON File  ──→  LangChain Documents  ──→  HuggingFace Embeddings  ──→  ChromaDB
                                              │
                                    ┌─────────┴──────────┐
                                    │ all-MiniLM-L6-v2   │
                                    │ 384-dim embeddings  │
                                    │ ~22M parameters     │
                                    │ Runs on CPU         │
                                    │ No API key needed   │
                                    └────────────────────┘

ChromaDB Configuration:
├── Collection Name:   knowledge_base
├── Distance Metric:   cosine similarity
├── Persistence:       Docker volume (chroma_db/)
├── Retriever Type:    similarity_score_threshold
├── Top-K:             3 documents
└── Score Threshold:   0.2 (minimum relevance)
```

**Embedding Model Choice:** We replaced `OpenAIEmbeddings(model='text-embedding-3-small')`
with `HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')` because:

- No API key required (runs locally)
- No per-token cost
- Fast on CPU (~384-dim vectors)
- Well-suited for semantic similarity retrieval
- Pre-downloaded during Docker build for zero cold-start latency

---

## 4. LangGraph Agent Workflow

### 4.1 Graph Topology

```
                    ┌─────────────────────┐
                    │     START (Entry)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  categorize_inquiry  │
                    │                     │
                    │  Input:  query text  │
                    │  Output: category    │
                    │  LLM:   structured   │
                    └──────────┬──────────┘
                               │ (always)
                    ┌──────────▼──────────────┐
                    │ analyze_inquiry_sentiment│
                    │                         │
                    │  Input:  query text      │
                    │  Output: sentiment       │
                    │  LLM:   structured       │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   determine_route    │
                    │  (Conditional Edge)  │
                    └──┬──┬──┬──┬─────────┘
                       │  │  │  │
          ┌────────────┘  │  │  └─────────────────┐
          │               │  │                    │
          ▼               ▼  ▼                    ▼
┌─────────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────────┐
│   generate_     │ │ generate_  │ │ generate_  │ │   escalate_to_   │
│   technical_    │ │ billing_   │ │ general_   │ │   human_agent    │
│   response      │ │ response   │ │ response   │ │                  │
│                 │ │            │ │            │ │  (no LLM call)   │
│  RAG + Claude   │ │ RAG+Claude │ │ RAG+Claude │ │  static message  │
└────────┬────────┘ └─────┬──────┘ └─────┬──────┘ └────────┬─────────┘
         │                │              │                  │
         └────────────────┴──────────────┴──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │        END          │
                    └─────────────────────┘
```

### 4.2 Routing Logic

```python
def determine_route(support_state) -> str:
    """Decision function for the conditional edge."""

    if sentiment == "Negative":
        return "escalate_to_human_agent"     # Priority: unhappy customers
    elif category == "Technical":
        return "generate_technical_response"  # RAG: technical collection
    elif category == "Billing":
        return "generate_billing_response"    # RAG: billing collection
    else:
        return "generate_general_response"    # RAG: general collection
```

**Design Rationale:** Negative sentiment always triggers escalation regardless of category.
This ensures frustrated customers are immediately routed to human support rather than
receiving an automated response that might worsen their experience.

### 4.3 State Transitions

```
Step 1: { customer_query: "..." }
                │
                ▼ categorize_inquiry
Step 2: { customer_query: "...", query_category: "Technical" }
                │
                ▼ analyze_inquiry_sentiment
Step 3: { customer_query: "...", query_category: "Technical", query_sentiment: "Neutral" }
                │
                ▼ generate_technical_response (via router)
Step 4: { customer_query: "...", query_category: "Technical",
          query_sentiment: "Neutral", final_response: "..." }
                │
                ▼ END
```

### 4.4 LLM Calls Per Query

| Step | LLM Call | Method | Tokens (approx) |
|------|----------|--------|-----------------|
| Categorize | 1 call | `with_structured_output(QueryCategory)` | ~200 input, ~10 output |
| Sentiment | 1 call | `with_structured_output(QuerySentiment)` | ~180 input, ~10 output |
| Generate Response | 1 call | `ChatPromptTemplate \| llm` | ~500 input, ~300 output |
| **Total** | **3 calls** | | **~1,200 tokens** |

*Escalation path uses 2 LLM calls (no generation call).*

---

## 5. Data Flow & Sequence Diagram

```
User            Chainlit        LangGraph         Claude API       ChromaDB
 │                │                │                  │               │
 │  Send query    │                │                  │               │
 │───────────────▶│                │                  │               │
 │                │  on_message()  │                  │               │
 │                │───────────────▶│                  │               │
 │                │                │                  │               │
 │                │                │  Categorize      │               │
 │                │                │─────────────────▶│               │
 │                │                │  {"Technical"}   │               │
 │                │                │◁─────────────────│               │
 │                │                │                  │               │
 │                │                │  Sentiment       │               │
 │                │                │─────────────────▶│               │
 │                │                │  {"Neutral"}     │               │
 │                │                │◁─────────────────│               │
 │                │                │                  │               │
 │                │                │  Router: Technical                │
 │                │                │                  │               │
 │                │                │  Retrieve docs   │               │
 │                │                │──────────────────────────────────▶│
 │                │                │  [doc1, doc2, doc3]              │
 │                │                │◁──────────────────────────────────│
 │                │                │                  │               │
 │                │                │  Generate        │               │
 │                │                │─────────────────▶│               │
 │                │                │  "Here is..."    │               │
 │                │                │◁─────────────────│               │
 │                │                │                  │               │
 │                │  final_state   │                  │               │
 │                │◁───────────────│                  │               │
 │                │                │                  │               │
 │                │  Format + Badge│                  │               │
 │  Response      │                │                  │               │
 │◁───────────────│                │                  │               │
 │                │                │                  │               │
```

---

## 6. RAG Pipeline Details

### 6.1 Ingestion (One-time on Startup)

```
router_agent_documents.json
        │
        ▼
┌───────────────────┐
│   JSON Parser     │
│                   │
│  Extract:         │
│  - text (content) │
│  - metadata       │
│    - category     │
│    - topic        │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ LangChain         │
│ Document()        │
│                   │
│ page_content=text │
│ metadata={...}    │
└────────┬──────────┘
         │  (batch)
         ▼
┌───────────────────┐       ┌──────────────────┐
│ HuggingFace       │──────▶│    ChromaDB      │
│ Embeddings        │       │                  │
│                   │       │  Collection:     │
│ Model: MiniLM-L6  │       │  knowledge_base  │
│ Dims:  384        │       │                  │
│ Device: CPU       │       │  Index: HNSW     │
│                   │       │  Metric: cosine  │
└───────────────────┘       └──────────────────┘
```

### 6.2 Retrieval (Per Query)

```
User Query: "What payment methods do you accept?"
        │
        ▼
┌───────────────────────────┐
│  Metadata Pre-Filter      │
│                           │
│  category == "billing"    │  ◁── Set dynamically based on
│  (from categorize_inquiry)│      classification result
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│  Embedding & Similarity   │
│                           │
│  1. Embed query → 384d    │
│  2. Cosine similarity     │
│  3. Filter: score ≥ 0.2   │
│  4. Top K = 3             │
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│  Retrieved Documents      │
│                           │
│  Doc1: "We accept Visa,   │
│         MasterCard..."    │
│  Doc2: "Subscription      │
│         plans include..." │
│  Doc3: "Invoices are      │
│         generated..."     │
└───────────────────────────┘
         │
         ▼
┌───────────────────────────┐
│  Prompt Assembly          │
│                           │
│  Template:                │
│  "Craft a clear billing   │
│   response for:           │
│   Query: {query}          │
│   KB Info: {docs}"        │
└────────┬──────────────────┘
         │
         ▼
┌───────────────────────────┐
│  Claude Generation        │
│                           │
│  Model: claude-sonnet-4   │
│  Temp:  0                 │
│  Max:   2048 tokens       │
└───────────────────────────┘
```

### 6.3 Metadata Filtering Strategy

The knowledge base uses a **category** field in document metadata to enable filtered retrieval:

| Query Classification | Metadata Filter | Effect |
|---------------------|----------------|--------|
| Technical | `{"category": "technical"}` | Only searches technical docs |
| Billing | `{"category": "billing"}` | Only searches billing docs |
| General | `{"category": "general"}` | Only searches general docs |

This is a **two-stage retrieval** approach: first filter by category, then rank by semantic
similarity within that category. This prevents billing questions from surfacing technical
docs and vice versa.

---

## 7. Technology Stack

### 7.1 Runtime Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                    Python 3.12 Runtime                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  AI / LLM                          Vector Store             │
│  ┌──────────────────────┐          ┌──────────────────┐    │
│  │ langchain >= 0.3.14  │          │ chromadb >= 0.5  │    │
│  │ langchain-anthropic  │          │ langchain-chroma │    │
│  │ langchain-community  │          └──────────────────┘    │
│  │ langgraph >= 0.2.64  │                                  │
│  └──────────────────────┘          Embeddings              │
│                                    ┌──────────────────┐    │
│  UI                                │ sentence-         │    │
│  ┌──────────────────────┐          │  transformers    │    │
│  │ chainlit >= 1.3.0    │          │ huggingface-hub  │    │
│  └──────────────────────┘          └──────────────────┘    │
│                                                             │
│  Utilities                                                  │
│  ┌──────────────────────┐                                  │
│  │ pydantic >= 2.0      │                                  │
│  │ python-dotenv        │                                  │
│  │ tqdm                 │                                  │
│  └──────────────────────┘                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 External Services

| Service | Provider | Purpose | Auth |
|---------|----------|---------|------|
| Claude API | Anthropic | LLM inference (classify, generate) | `ANTHROPIC_API_KEY` |
| HuggingFace Hub | HuggingFace | Download embedding model (first run) | None (public model) |

### 7.3 Infrastructure

| Component | Local (Docker Desktop) | AWS EC2 |
|-----------|----------------------|---------|
| Container Runtime | Docker Desktop (Windows) | Docker CE (Ubuntu) |
| Orchestration | docker-compose | docker-compose |
| Reverse Proxy | N/A (direct port) | Nginx |
| SSL/TLS | N/A | Let's Encrypt (optional) |
| Persistent Storage | Docker named volume | Docker named volume |
| OS | Windows host / Linux container | Ubuntu 22.04/24.04 |

---

## 8. File Structure & Module Map

```
customer-support-agent/
│
├── app.py                              ◁── ENTRY POINT (Chainlit server)
│   ├── Imports: agent.py, knowledge_base_loader.py
│   ├── Global: Builds agent on startup
│   ├── on_chat_start()     → Session init
│   ├── on_message()        → Agent pipeline execution
│   ├── run_agent_sync()    → Sync wrapper for LangGraph
│   ├── on_chat_end()       → Cleanup
│   └── on_stop()           → Interrupt handler
│
├── agent.py                            ◁── CORE LOGIC (LangGraph workflow)
│   ├── CustomerSupportState   (TypedDict)
│   ├── QueryCategory          (Pydantic model)
│   ├── QuerySentiment         (Pydantic model)
│   ├── get_llm()              → ChatAnthropic instance
│   ├── get_kbase_retriever()  → ChromaDB retriever
│   └── build_support_agent()  → Compiled StateGraph
│       ├── categorize_inquiry()
│       ├── analyze_inquiry_sentiment()
│       ├── generate_technical_response()
│       ├── generate_billing_response()
│       ├── generate_general_response()
│       ├── escalate_to_human_agent()
│       └── determine_route()
│
├── knowledge_base_loader.py            ◁── DATA LAYER
│   ├── get_embedding_model()  → HuggingFaceEmbeddings
│   └── load_and_index_knowledge_base() → Chroma
│
├── data/
│   └── router_agent_documents.json     ◁── KNOWLEDGE BASE
│       └── Array of { text, metadata: { category, topic } }
│
├── .chainlit/
│   └── config.toml                     ◁── UI CONFIG (theme, features)
│
├── chainlit.md                         ◁── WELCOME PAGE CONTENT
│
├── Dockerfile                          ◁── CONTAINER IMAGE
├── docker-compose.yml                  ◁── LOCAL ORCHESTRATION
├── .env.example                        ◁── ENV TEMPLATE
├── .dockerignore                       ◁── BUILD EXCLUSIONS
├── .gitignore                          ◁── VCS EXCLUSIONS
├── requirements.txt                    ◁── PYTHON DEPS
│
├── aws/
│   ├── deploy.sh                       ◁── EC2 AUTOMATED DEPLOYMENT
│   └── nginx.conf                      ◁── REVERSE PROXY CONFIG
│
├── public/
│   └── custom.css                      ◁── OPTIONAL UI STYLES
│
├── README.md                           ◁── USER GUIDE
└── ARCHITECTURE.md                     ◁── THIS FILE
```

### Module Dependency Graph

```
app.py
 ├──▶ agent.py
 │     ├──▶ langchain-anthropic  (ChatAnthropic)
 │     ├──▶ langchain-core       (prompts, documents)
 │     ├──▶ langgraph            (StateGraph, END, MemorySaver)
 │     └──▶ pydantic             (BaseModel for structured output)
 │
 ├──▶ knowledge_base_loader.py
 │     ├──▶ langchain-community  (HuggingFaceEmbeddings)
 │     ├──▶ langchain-chroma     (Chroma)
 │     ├──▶ langchain-core       (Document)
 │     └──▶ sentence-transformers (underlying embedding engine)
 │
 └──▶ chainlit                   (UI framework)
```

---

## 9. Deployment Architecture

### 9.1 Local Deployment (Docker Desktop — Windows)

```
┌──────────────────────────────────────────────────┐
│                Windows Host Machine               │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │           Docker Desktop (WSL2)             │  │
│  │                                            │  │
│  │  ┌──────────────────────────────────────┐  │  │
│  │  │  Container: customer-support-agent    │  │  │
│  │  │                                      │  │  │
│  │  │  ┌──────────┐    ┌───────────────┐  │  │  │
│  │  │  │ Chainlit │    │  LangGraph    │  │  │  │
│  │  │  │  :8000   │    │  Agent        │  │  │  │
│  │  │  └──────────┘    └───────────────┘  │  │  │
│  │  │                                      │  │  │
│  │  │  ┌──────────┐    ┌───────────────┐  │  │  │
│  │  │  │ ChromaDB │    │  Embeddings   │  │  │  │
│  │  │  │ (in-proc)│    │  (in-proc)    │  │  │  │
│  │  │  └──────────┘    └───────────────┘  │  │  │
│  │  │       │                              │  │  │
│  │  │       ▼                              │  │  │
│  │  │  ┌──────────────────┐               │  │  │
│  │  │  │ Docker Volume    │               │  │  │
│  │  │  │ chroma_data      │               │  │  │
│  │  │  └──────────────────┘               │  │  │
│  │  └──────────────────────────────────────┘  │  │
│  │                                            │  │
│  │  Port Mapping: localhost:8000 → :8000      │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  Browser → http://localhost:8000                  │
└──────────────────────────────────────────────────┘
         │
         │ HTTPS (outbound)
         ▼
┌──────────────────┐
│ Anthropic API    │
│ api.anthropic.com│
└──────────────────┘
```

### 9.2 AWS EC2 Deployment (Ubuntu)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AWS VPC                                      │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  EC2 Instance (Ubuntu)                         │  │
│  │                  t3.medium (2 vCPU, 4GB RAM)                  │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │                    Nginx                                 │  │  │
│  │  │                    :80 / :443                             │  │  │
│  │  │                                                         │  │  │
│  │  │  ┌─────────────┐   ┌─────────────────────────────────┐ │  │  │
│  │  │  │ SSL/TLS     │   │ Reverse Proxy                   │ │  │  │
│  │  │  │ (optional   │──▶│ proxy_pass → localhost:8000     │ │  │  │
│  │  │  │ LetsEncrypt)│   │ WebSocket upgrade support       │ │  │  │
│  │  │  └─────────────┘   └──────────────┬──────────────────┘ │  │  │
│  │  └────────────────────────────────────┼─────────────────────┘  │  │
│  │                                       │                        │  │
│  │  ┌────────────────────────────────────┼─────────────────────┐  │  │
│  │  │         Docker Engine              │                     │  │  │
│  │  │                                    │                     │  │  │
│  │  │  ┌─────────────────────────────────▼──────────────────┐ │  │  │
│  │  │  │  Container: customer-support-agent                  │ │  │  │
│  │  │  │                                                    │ │  │  │
│  │  │  │  Chainlit :8000  ←→  LangGraph  ←→  ChromaDB      │ │  │  │
│  │  │  │                                                    │ │  │  │
│  │  │  └────────────────────────────────────────────────────┘ │  │  │
│  │  │                                                         │  │  │
│  │  │  ┌────────────────────┐                                │  │  │
│  │  │  │  Volume: chroma_db │                                │  │  │
│  │  │  └────────────────────┘                                │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │                                                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Security Group Rules:                                              │
│  ├── Inbound:  TCP 22 (SSH), TCP 80 (HTTP), TCP 443 (HTTPS)       │
│  └── Outbound: All traffic (for API calls)                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
         │
         │ HTTPS (outbound)
         ▼
┌──────────────────┐     ┌──────────────────────┐
│ Anthropic API    │     │ HuggingFace Hub      │
│ api.anthropic.com│     │ (first run only)     │
└──────────────────┘     └──────────────────────┘
```

### 9.3 Container Internals

```
Docker Image Layers:
┌────────────────────────────────────────────┐
│  CMD ["chainlit", "run", "app.py", ...]    │  ◁── Runtime
├────────────────────────────────────────────┤
│  COPY app.py agent.py *.py data/ ...       │  ◁── App code
├────────────────────────────────────────────┤
│  RUN python -c "SentenceTransformer(...)"  │  ◁── Cached model (~90MB)
├────────────────────────────────────────────┤
│  RUN pip install -r requirements.txt       │  ◁── Python deps (~1.2GB)
├────────────────────────────────────────────┤
│  RUN apt-get install build-essential ...   │  ◁── System deps
├────────────────────────────────────────────┤
│  FROM python:3.12-slim                     │  ◁── Base image (~150MB)
└────────────────────────────────────────────┘

Volumes:
├── chroma_data (named volume) → /app/chroma_db   (persistent index)
└── ./data (bind mount, read-only) → /app/data     (knowledge base JSON)
```

---

## 10. Security & Configuration

### 10.1 Secret Management

```
Environment Variables:
├── ANTHROPIC_API_KEY    →  Required. Stored in .env (git-ignored)
├── DATA_DIR             →  Optional. Default: ./data
└── CHROMA_PERSIST_DIR   →  Optional. Default: ./chroma_db

Secret Flow:
.env file → docker-compose.yml (env_file) → Container environment → os.environ
```

### 10.2 Network Security

```
Local Deployment:
├── Exposed: localhost:8000 only (no external access)
├── Outbound: HTTPS to api.anthropic.com
└── No authentication (single-user local dev)

AWS Deployment:
├── Nginx: Terminates HTTP/HTTPS, proxies to localhost:8000
├── Container: Binds to localhost only (not externally reachable)
├── Security Group: Restricts inbound to 22, 80, 443
├── Optional: TLS via Let's Encrypt
└── Optional: Add Chainlit auth for multi-user
```

### 10.3 Data Privacy

- Knowledge base JSON is mounted read-only in the container
- ChromaDB vectors are stored in an isolated Docker volume
- No user queries are logged to disk (in-memory LangGraph state only)
- Anthropic API calls follow Anthropic's data retention policies
- Session data lives only in process memory and expires on disconnect

---

## 11. Scalability Considerations

### 11.1 Current Limitations (Single Container)

| Dimension | Current Limit | Bottleneck |
|-----------|--------------|------------|
| Concurrent users | ~20-50 | Chainlit async capacity |
| Queries/second | ~3-5 | Claude API latency (~2-3s per call, 3 calls per query) |
| Knowledge base size | ~10K docs | ChromaDB in-process memory |
| Vector dimensions | 384 (MiniLM) | Fixed by model choice |

### 11.2 Scaling Strategies

```
Vertical Scaling (Quick Wins):
├── Larger EC2 instance (t3.xlarge → m5.2xlarge)
├── Switch to faster Claude model (claude-haiku for classify/sentiment)
├── Enable GPU for embeddings (faster indexing)
└── Increase ChromaDB cache size

Horizontal Scaling (Production):
├── Load balancer (ALB) → Multiple container instances
├── External ChromaDB server (Chroma Cloud or self-hosted)
├── Redis for session state (replace MemorySaver)
├── Separate embedding service (GPU instance)
└── CDN for static assets (CloudFront)

Cost Optimization:
├── Use claude-haiku for classification + sentiment (cheaper, faster)
├── Use claude-sonnet for response generation only
├── Cache frequent queries (Redis)
├── Batch embedding operations
└── Reserved instances for EC2
```

### 11.3 Potential Production Architecture

```
┌──────────┐     ┌─────────┐     ┌──────────────────────┐
│ CloudFront│────▶│   ALB   │────▶│  ECS Fargate Cluster │
│   (CDN)   │     │         │     │                      │
└──────────┘     └─────────┘     │  ┌─────┐  ┌─────┐   │
                                  │  │App 1│  │App 2│   │
                                  │  └──┬──┘  └──┬──┘   │
                                  └─────┼────────┼──────┘
                                        │        │
                                  ┌─────▼────────▼──────┐
                                  │   ElastiCache        │
                                  │   (Redis - sessions) │
                                  └──────────┬───────────┘
                                             │
                                  ┌──────────▼───────────┐
                                  │  Chroma Cloud /      │
                                  │  OpenSearch           │
                                  │  (Vector Store)       │
                                  └──────────────────────┘
```

---

*Document Version: 1.0 | Generated for Customer Support Router Agentic RAG System*
