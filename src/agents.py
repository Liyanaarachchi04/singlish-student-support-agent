import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

try:
    from src.rag_pipeline import query_rag_system
except ModuleNotFoundError:
    from rag_pipeline import query_rag_system


# ==========================================
# 1. STRUCTURED SCHEMAS FOR AGENT MESSAGES
# ==========================================

class RouterOutput(BaseModel):
    intent: str = Field(description="Detected intent: 'academic_query', 'research_benchmark', or 'general'")
    requires_rag: bool = Field(description="True if context lookup in university documents is needed")
    clean_query: str = Field(description="Cleaned/normalized query text")

class CritiqueOutput(BaseModel):
    singlish_tokens: List[str] = Field(description="Identified Singlish or code-mixed terms in the input")
    tokenization_issue: str = Field(description="Explanation of potential tokenization or structural parsing breaking points")
    estimated_intent_confidence: float = Field(description="Confidence score between 0.0 and 1.0 for processing this query")

class FinalResponse(BaseModel):
    user_query: str
    detected_intent: str
    rag_context_used: List[str]
    linguistic_critique: Dict[str, Any]
    final_answer: str


# ==========================================
# 2. AGENT INITIALIZATION (GROQ ENGINE)
# ==========================================

def get_groq_llm(model_name: str = "llama-3.1-8b-instant", temp: float = 0.0):
    """Returns Groq model with robust API key check."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    return ChatGroq(model_name=model_name, groq_api_key=api_key, temperature=temp)


# ==========================================
# 3. AGENT DEFINITIONS
# ==========================================

def agent_1_intent_router(query: str) -> RouterOutput:
    """Agent 1: Fast Intent Classifier using Groq Llama 3.1 8B (Router Pattern)."""
    llm = get_groq_llm("llama-3.1-8b-instant", temp=0.0)
    parser = JsonOutputParser(pydantic_object=RouterOutput)

    prompt = ChatPromptTemplate.from_template(
        "You are an Intent Classification Agent for a Sri Lankan University Support System.\n"
        "Analyze the following student query (which may be in code-mixed Singlish / English):\n"
        "Query: {query}\n\n"
        "Determine if this is an 'academic_query' (related to exams, registration, forms, handbooks),\n"
        "'research_benchmark' (analyzing Singlish AI model failures), or 'general'.\n"
        "{format_instructions}"
    )

    chain = prompt | llm | parser
    res = chain.invoke({"query": query, "format_instructions": parser.get_format_instructions()})
    return RouterOutput(**res)


def agent_2_rag_retriever(query: str) -> List[str]:
    """Agent 2: Tool-Use / ReAct Retriever querying Chroma vector store."""
    docs = query_rag_system(query, k=2)
    return [d.page_content for d in docs]


def agent_3_reflection_critique(query: str) -> CritiqueOutput:
    """Agent 3: Reflection Agent using Groq Llama 3.3 70B."""
    llm = get_groq_llm("llama-3.3-70b-versatile", temp=0.1)
    parser = JsonOutputParser(pydantic_object=CritiqueOutput)

    prompt = ChatPromptTemplate.from_template(
        "You are an NLP Research Reflection Agent specializing in Sri Lankan code-mixed text (Singlish).\n"
        "Analyze the following input query for potential model tokenization breakdown:\n"
        "Input: {query}\n\n"
        "Identify specific Singlish words (e.g., 'karanne', 'monawada', 'koheda'), explain structural parsing challenges,\n"
        "and provide an intent confidence score.\n"
        "{format_instructions}"
    )

    chain = prompt | llm | parser
    res = chain.invoke({"query": query, "format_instructions": parser.get_format_instructions()})
    return CritiqueOutput(**res)


def agent_4_final_synthesizer(query: str, router_res: RouterOutput, rag_contexts: List[str], critique_res: CritiqueOutput) -> str:
    """Agent 4: Synthesizer model providing final response using Groq Llama 3.3 70B."""
    llm = get_groq_llm("llama-3.3-70b-versatile", temp=0.3)

    context_str = "\n---\n".join(rag_contexts) if rag_contexts else "No external handbook context required."

    prompt = ChatPromptTemplate.from_template(
        "You are an AI Student Support Assistant at Horizon Campus Sri Lanka.\n"
        "Respond clearly and accurately to the student query in helpful, professional English,\n"
        "while referencing Sinhala terms where appropriate.\n\n"
        "User Query: {query}\n"
        "Detected Intent: {intent}\n"
        "University Handbook Context:\n{context}\n"
        "Linguistic Critique Notes: {critique}\n\n"
        "Provide a complete, direct, and polite answer to the student."
    )

    chain = prompt | llm
    res = chain.invoke({
        "query": query,
        "intent": router_res.intent,
        "context": context_str,
        "critique": critique_res.tokenization_issue
    })
    return res.content


# ==========================================
# 4. MULTI-AGENT ORCHESTRATION WORKFLOW
# ==========================================

def run_multi_agent_pipeline(user_query: str) -> FinalResponse:
    """Orchestrates structured agent-to-agent communication pipeline."""
    # Step 1: Agent 1 - Intent Classification
    router_data = agent_1_intent_router(user_query)

    # Step 2: Agent 2 - RAG Retrieval (if required)
    rag_contexts = []
    if router_data.requires_rag:
        rag_contexts = agent_2_rag_retriever(user_query)

    # Step 3: Agent 3 - Reflection & Singlish Linguistic Critique
    critique_data = agent_3_reflection_critique(user_query)

    # Step 4: Agent 4 - Final Response Synthesis
    final_text = agent_4_final_synthesizer(
        query=user_query,
        router_res=router_data,
        rag_contexts=rag_contexts,
        critique_res=critique_data
    )

    return FinalResponse(
        user_query=user_query,
        detected_intent=router_data.intent,
        rag_context_used=rag_contexts,
        linguistic_critique=critique_data.model_dump(),
        final_answer=final_text
    )