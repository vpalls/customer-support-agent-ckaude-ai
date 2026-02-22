"""
LangGraph Customer Support Agent - Claude Edition
Converts the original OpenAI-based agent to use Anthropic Claude.
"""
import os
from typing import TypedDict, Literal
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  State & Schema Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CustomerSupportState(TypedDict):
    customer_query: str
    query_category: str
    query_sentiment: str
    final_response: str


class QueryCategory(BaseModel):
    categorized_topic: Literal['Technical', 'Billing', 'General']


class QuerySentiment(BaseModel):
    sentiment: Literal['Positive', 'Neutral', 'Negative']


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM Initialization — Claude
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_llm():
    """Initialize Claude LLM via LangChain."""
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        max_tokens=2048,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Retriever Helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_kbase_retriever(kbase_db):
    """Create a retriever from the ChromaDB vector store."""
    return kbase_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.2},
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Node Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_support_agent(kbase_search):
    """Build and compile the LangGraph customer support agent."""

    llm = get_llm()

    # ── Node 1: Categorize Inquiry ──
    def categorize_inquiry(support_state: CustomerSupportState) -> CustomerSupportState:
        """Classify the customer query into Technical, Billing, or General."""
        query = support_state["customer_query"]
        ROUTE_CATEGORY_PROMPT = """Act as a customer support agent trying to best categorize the customer query.
You are an agent for an AI products and hardware company.

Please read the customer query below and determine the best category from the following list:
'Technical', 'Billing', or 'General'.

Remember:
- Technical queries will focus more on technical aspects like AI models, hardware, software related queries etc.
- General queries will focus more on general aspects like contacting support, finding things, policies etc.
- Billing queries will focus more on payment and purchase related aspects

Return just the category name (from one of the above)

Query:
{customer_query}"""
        prompt = ROUTE_CATEGORY_PROMPT.format(customer_query=query)
        route_category = llm.with_structured_output(QueryCategory).invoke(prompt)
        print(f"  📂 Category: {route_category.categorized_topic}")
        return {"query_category": route_category.categorized_topic}

    # ── Node 2: Analyze Sentiment ──
    def analyze_inquiry_sentiment(support_state: CustomerSupportState) -> CustomerSupportState:
        """Analyze the sentiment of the customer query."""
        query = support_state["customer_query"]
        SENTIMENT_PROMPT = """Act as a customer support agent trying to best categorize the customer query's sentiment.
You are an agent for an AI products and hardware company.

Please read the customer query below, analyze its sentiment which should be one from the following list:
'Positive', 'Neutral', or 'Negative'.

Return just the sentiment (from one of the above)

Query:
{customer_query}"""
        prompt = SENTIMENT_PROMPT.format(customer_query=query)
        sentiment_category = llm.with_structured_output(QuerySentiment).invoke(prompt)
        print(f"  🎭 Sentiment: {sentiment_category.sentiment}")
        return {"query_sentiment": sentiment_category.sentiment}

    # ── Node 3: Technical Response ──
    def generate_technical_response(support_state: CustomerSupportState) -> CustomerSupportState:
        """Provide a technical support response using RAG."""
        categorized_topic = support_state["query_category"]
        query = support_state["customer_query"]

        if categorized_topic.lower() == "technical":
            metadata_filter = {"category": "technical"}
            kbase_search.search_kwargs["filter"] = metadata_filter
            relevant_docs = kbase_search.invoke(query)
            retrieved_content = "\n\n".join(doc.page_content for doc in relevant_docs)

            prompt = ChatPromptTemplate.from_template("""
Craft a clear and detailed technical support response for the following customer query.
Use the provided knowledge base information to enrich your response.
In case there is no knowledge base information or you do not know the answer just say:
Apologies I was not able to answer your question, please reach out to +1-xxx-xxxx

Customer Query:
{customer_query}

Relevant Knowledge Base Information:
{retrieved_content}
""")
            chain = prompt | llm
            tech_reply = chain.invoke({
                "customer_query": query,
                "retrieved_content": retrieved_content,
            }).content
        else:
            tech_reply = "Apologies I was not able to answer your question, please reach out to +1-xxx-xxxx"

        print(f"  🔧 Technical response generated")
        return {"final_response": tech_reply}

    # ── Node 4: Billing Response ──
    def generate_billing_response(support_state: CustomerSupportState) -> CustomerSupportState:
        """Provide a billing support response using RAG."""
        categorized_topic = support_state["query_category"]
        query = support_state["customer_query"]

        if categorized_topic.lower() == "billing":
            metadata_filter = {"category": "billing"}
            kbase_search.search_kwargs["filter"] = metadata_filter
            relevant_docs = kbase_search.invoke(query)
            retrieved_content = "\n\n".join(doc.page_content for doc in relevant_docs)

            prompt = ChatPromptTemplate.from_template("""
Craft a clear and detailed billing support response for the following customer query.
Use the provided knowledge base information to enrich your response.
In case there is no knowledge base information or you do not know the answer just say:
Apologies I was not able to answer your question, please reach out to +1-xxx-xxxx

Customer Query:
{customer_query}

Relevant Knowledge Base Information:
{retrieved_content}
""")
            chain = prompt | llm
            billing_reply = chain.invoke({
                "customer_query": query,
                "retrieved_content": retrieved_content,
            }).content
        else:
            billing_reply = "Apologies I was not able to answer your question, please reach out to +1-xxx-xxxx"

        print(f"  💳 Billing response generated")
        return {"final_response": billing_reply}

    # ── Node 5: General Response ──
    def generate_general_response(support_state: CustomerSupportState) -> CustomerSupportState:
        """Provide a general support response using RAG."""
        categorized_topic = support_state["query_category"]
        query = support_state["customer_query"]

        if categorized_topic.lower() == "general":
            metadata_filter = {"category": "general"}
            kbase_search.search_kwargs["filter"] = metadata_filter
            relevant_docs = kbase_search.invoke(query)
            retrieved_content = "\n\n".join(doc.page_content for doc in relevant_docs)

            prompt = ChatPromptTemplate.from_template("""
Craft a clear and detailed general support response for the following customer query.
Use the provided knowledge base information to enrich your response.
In case there is no knowledge base information or you do not know the answer just say:
Apologies I was not able to answer your question, please reach out to +1-xxx-xxxx

Customer Query:
{customer_query}

Relevant Knowledge Base Information:
{retrieved_content}
""")
            chain = prompt | llm
            general_reply = chain.invoke({
                "customer_query": query,
                "retrieved_content": retrieved_content,
            }).content
        else:
            general_reply = "Apologies I was not able to answer your question, please reach out to +1-xxx-xxxx"

        print(f"  📋 General response generated")
        return {"final_response": general_reply}

    # ── Node 6: Escalate ──
    def escalate_to_human_agent(support_state: CustomerSupportState) -> CustomerSupportState:
        """Escalate the query to a human agent if sentiment is negative."""
        print(f"  🚨 Escalating to human agent")
        return {
            "final_response": (
                "😟 **We're sorry to hear you're having trouble!**\n\n"
                "Your concern has been flagged as **high priority** and "
                "someone from our team will be reaching out to you shortly.\n\n"
                "📞 In the meantime, you can also reach us at: **1-800-123-4567**\n"
                "📧 Or email: **priority-support@example.com**"
            )
        }

    # ── Router ──
    def determine_route(support_state: CustomerSupportState) -> str:
        """Route the inquiry based on sentiment and category."""
        if support_state["query_sentiment"] == "Negative":
            return "escalate_to_human_agent"
        elif support_state["query_category"] == "Technical":
            return "generate_technical_response"
        elif support_state["query_category"] == "Billing":
            return "generate_billing_response"
        else:
            return "generate_general_response"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  Build the LangGraph Workflow
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    graph = StateGraph(CustomerSupportState)

    # Add nodes
    graph.add_node("categorize_inquiry", categorize_inquiry)
    graph.add_node("analyze_inquiry_sentiment", analyze_inquiry_sentiment)
    graph.add_node("generate_technical_response", generate_technical_response)
    graph.add_node("generate_billing_response", generate_billing_response)
    graph.add_node("generate_general_response", generate_general_response)
    graph.add_node("escalate_to_human_agent", escalate_to_human_agent)

    # Add edges
    graph.add_edge("categorize_inquiry", "analyze_inquiry_sentiment")
    graph.add_conditional_edges(
        "analyze_inquiry_sentiment",
        determine_route,
        [
            "generate_technical_response",
            "generate_billing_response",
            "generate_general_response",
            "escalate_to_human_agent",
        ],
    )

    # Terminal edges
    graph.add_edge("generate_technical_response", END)
    graph.add_edge("generate_billing_response", END)
    graph.add_edge("generate_general_response", END)
    graph.add_edge("escalate_to_human_agent", END)

    # Entry point
    graph.set_entry_point("categorize_inquiry")

    # Compile
    memory = MemorySaver()
    compiled_agent = graph.compile(checkpointer=memory)
    print("✅ LangGraph agent compiled successfully")
    return compiled_agent
