# 🤖 AI Customer Support Agent

Welcome to the **Intelligent Customer Support Router** — an Agentic RAG system powered by **Claude AI** and **LangGraph**.

## How It Works

This agent uses a multi-step pipeline to handle your support queries:

1. **🏷️ Query Classification** — Your question is categorized as *Technical*, *Billing*, or *General*
2. **🎭 Sentiment Analysis** — We detect if you're *Positive*, *Neutral*, or *Negative*
3. **🔀 Smart Routing** — Based on category and sentiment:
   - 🔧 **Technical queries** → Technical knowledge base + AI response
   - 💳 **Billing queries** → Billing knowledge base + AI response
   - 📋 **General queries** → General knowledge base + AI response
   - 🚨 **Negative sentiment** → Automatic escalation to human agent

## Try These Examples

- *"Do you support pre-trained models?"*
- *"What payment methods do you accept?"*
- *"What is your refund policy?"*
- *"I'm fed up with this faulty hardware, I need a refund!"*
- *"What are your working hours?"*

## Architecture

| Component | Technology |
|-----------|-----------|
| **LLM** | Claude (Anthropic) |
| **Framework** | LangGraph + LangChain |
| **Vector Store** | ChromaDB |
| **Embeddings** | sentence-transformers |
| **UI** | Chainlit |
| **Deployment** | Docker |

---

*Type your question below to get started!* 👇
