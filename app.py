# Streamlit Dashboard App Interface
import os
import streamlit as st
from src.agents import run_multi_agent_pipeline
from src.rag_pipeline import query_rag_system

# Page Configuration
st.set_page_config(
    page_title="Singlish AI Student Support & Benchmarking",
    page_icon="🎓",
    layout="wide"
)

# Initialize Secrets / Environment Variables
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "OPENROUTER_API_KEY" in st.secrets:
    os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

# Header Section
st.title("🎓 Intelligent Student Support & Singlish Benchmarking System")
st.markdown(
    "*Evaluating Multilingual AI Chatbots for Student Support Services in Sri Lankan Universities (IT41043 Research Tool)*"
)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Agentic Architecture")
    st.info(
        "**Multi-Agent Workflow & Routing:**\n"
        "- **Agent 1 (Groq - Llama 3.1 8B):** Fast Intent Classification (Router Pattern)\n"
        "- **Agent 2 (Chroma DB):** Academic Document Retrieval (ReAct Tool-Use Pattern)\n"
        "- **Agent 3 (OpenRouter - Claude 3.5):** Singlish Tokenization & Error Reflection (Self-Critique Pattern)\n"
        "- **Agent 4 (OpenRouter - Claude 3.5):** Final Synthesis"
    )
    st.markdown("---")
    st.markdown("**Module:** IT41043 Intelligent Systems")
    st.markdown("**Student Index:** ITBIN-2313-0056")

# Tab Layout
tab1, tab2 = st.tabs(["💬 Student Support Assistant", "🔬 Research Benchmarking Tool"])

# ==========================================
# TAB 1: STUDENT SUPPORT ASSISTANT (RAG UI)
# ==========================================
with tab1:
    st.subheader("Ask Academic Queries in Code-Mixed Singlish / English")
    st.caption("Example: *'Repeat exam register karanne kohomada form A2 ganna codehada?'* or *'IT41043 prerequisite subjects monawada?'*")

    user_query = st.text_input("Enter your query below:", placeholder="Type your query here...")

    if st.button("Submit Query", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Processing query across Multi-Agent Pipeline..."):
                try:
                    response = run_multi_agent_pipeline(user_query)

                    # Display Final Answer
                    st.success("### Response")
                    st.write(response.final_answer)

                    st.markdown("---")
                    
                    # Display Agent Breakdown
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🎯 Detected Intent")
                        st.code(f"Intent: {response.detected_intent}")

                        st.markdown("#### 📚 Retrieved Handbook Context")
                        if response.rag_context_used:
                            for idx, ctx in enumerate(response.rag_context_used, 1):
                                st.info(f"**Chunk {idx}:** {ctx}")
                        else:
                            st.write("No handbook search was required for this query.")

                    with col2:
                        st.markdown("#### 🔍 Singlish Linguistic Critique (Agent 3)")
                        critique = response.linguistic_critique
                        st.json(critique)

                except Exception as e:
                    st.error(f"An error occurred while executing agents: {str(e)}")
                    st.warning("Ensure API keys (GROQ_API_KEY and OPENROUTER_API_KEY) are configured in .streamlit/secrets.toml")

# ==========================================
# TAB 2: RESEARCH BENCHMARKING TOOL
# ==========================================
with tab2:
    st.subheader("Singlish NLP Model Breakdown & Tokenization Analyzer")
    st.markdown(
        "This tool evaluates how LLM tokenizers and intent classifiers process code-mixed Sinhala-English academic text."
    )

    benchmark_text = st.text_area(
        "Input Singlish Sample Query for Evaluation:",
        value="Semester registration late fee kiyada and medical MC submit karanne kata da?"
    )

    if st.button("Run Model Evaluation"):
        with st.spinner("Analyzing tokenization and structural breakdown..."):
            try:
                response = run_multi_agent_pipeline(benchmark_text)
                
                critique = response.linguistic_critique
                
                m1, m2 = st.columns(2)
                m1.metric("Detected Intent", response.detected_intent.upper())
                m2.metric("Parsing Confidence Score", f"{critique.get('estimated_intent_confidence', 0.0) * 100:.1f}%")

                st.markdown("### Identified Singlish / Code-Mixed Tokens")
                st.write(critique.get("singlish_tokens", []))

                st.markdown("### Structural Parsing Analysis")
                st.warning(critique.get("tokenization_issue", "No tokenization issue identified."))

            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
