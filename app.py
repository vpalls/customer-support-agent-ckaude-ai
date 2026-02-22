"""
Customer Support Router Agentic RAG System - Chainlit UI
Powered by Claude (Anthropic) + LangGraph + ChromaDB
"""
import chainlit as cl
import asyncio
import time
from agent import build_support_agent, get_kbase_retriever
from knowledge_base_loader import load_and_index_knowledge_base

# ──────────────────────────────────────────────
#  Global: Build agent once on server startup
# ──────────────────────────────────────────────
print("🚀 Initializing Customer Support Agent...")
kbase_db = load_and_index_knowledge_base()
kbase_search = get_kbase_retriever(kbase_db)
compiled_agent = build_support_agent(kbase_search)
print("✅ Agent ready!")


# ──────────────────────────────────────────────
#  Chainlit Lifecycle Hooks
# ──────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Initialize a new user session."""
    session_id = cl.user_session.get("id")
    cl.user_session.set("session_id", session_id)

    # Welcome message with rich formatting
    elements = [
        cl.Text(
            name="capabilities",
            content="""### 🛠️ What I Can Help With

| Category | Examples |
|----------|----------|
| **🔧 Technical** | AI models, hardware, software, integrations |
| **💳 Billing** | Payments, invoices, refunds, subscriptions |
| **📋 General** | Policies, support hours, shipping, contact info |

> 💡 **Tip:** I analyze your query's sentiment too — if you're frustrated, I'll escalate to a human agent!
""",
            display="inline",
        )
    ]

    await cl.Message(
        content="👋 **Welcome to AI Customer Support!**\n\n"
                "I'm your intelligent support assistant powered by **Claude AI**. "
                "I can route your questions to the right department and provide "
                "accurate answers from our knowledge base.\n\n"
                "**How can I help you today?**",
        elements=elements,
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Process each user message through the agentic RAG pipeline."""

    session_id = cl.user_session.get("session_id")

    # Step indicators for rich UX
    async with cl.Step(name="🏷️ Categorizing Query", type="tool") as step:
        step.input = message.content
        step.output = "Analyzing query category..."

    async with cl.Step(name="🎭 Analyzing Sentiment", type="tool") as step:
        step.input = message.content
        step.output = "Detecting sentiment..."

    # Run the agent in a thread to avoid blocking
    agent_result = await asyncio.to_thread(
        run_agent_sync, compiled_agent, message.content, session_id
    )

    # Extract results
    category = agent_result.get("query_category", "Unknown")
    sentiment = agent_result.get("query_sentiment", "Unknown")
    response = agent_result.get("final_response", "Sorry, I couldn't process your request.")

    # Sentiment emoji mapping
    sentiment_emoji = {
        "Positive": "😊",
        "Neutral": "😐",
        "Negative": "😟"
    }

    # Category emoji mapping
    category_emoji = {
        "Technical": "🔧",
        "Billing": "💳",
        "General": "📋"
    }

    # Build metadata badge
    badge = (
        f"\n\n---\n"
        f"*{category_emoji.get(category, '📌')} Category: **{category}** | "
        f"{sentiment_emoji.get(sentiment, '❓')} Sentiment: **{sentiment}***"
    )

    # Send the response
    await cl.Message(
        content=response + badge,
        author="Support Agent",
    ).send()


def run_agent_sync(agent, query: str, session_id: str) -> dict:
    """Run the LangGraph agent synchronously (called from async context)."""
    result = {}
    events = agent.stream(
        {"customer_query": query},
        {"configurable": {"thread_id": session_id}},
        stream_mode="values",
    )
    for event in events:
        result = event
    return result


@cl.on_chat_end
async def on_chat_end():
    """Cleanup on session end."""
    print(f"Session ended: {cl.user_session.get('session_id')}")


@cl.on_stop
async def on_stop():
    """Handle user stopping a response."""
    await cl.Message(content="⏹️ Response stopped.").send()
