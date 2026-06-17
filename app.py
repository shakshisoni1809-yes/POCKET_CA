import json
import os
import datetime
import chromadb
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema import messages_from_dict, messages_to_dict

# ── STREAMLIT PAGE CONFIG ────────────────────────────────
st.set_page_config(
    page_title="PocketCA",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #061B34; color: white; }
[data-testid="stSidebar"] { background-color: #1E2F4F; }
h1, h2, h3 { color: #D9D9D6; font-weight: 700; }
p, label, span, div { color: white; }
.stButton > button {
    background-color: #496584; color: white; border-radius: 12px;
    border: none; width: 100%; padding: 12px; font-weight: 600;
}
.stButton > button:hover { background-color: #7C92AF; color: #061B34; }
.stTextInput input, .stTextArea textarea {
    background-color: #1E2F4F; color: white; border: 1px solid #496584; border-radius: 10px;
}
.stChatMessage { background-color: #1E2F4F; border-radius: 12px; padding: 10px; margin-bottom: 10px; }
.stCaption { color: #D9D9D6; }
</style>
""", unsafe_allow_html=True)

GROQ_API_KEY = "YOUR_GROQ_API_KEY"

# ── SYSTEM PROMPT ────────────────────────────────────────
SYSTEM_PROMPT = """You are the Pocket CA Agent, a highly precise, reliable, and brutally honest AI financial assistant.
Your primary function is to act as an expert Chartered Accountant, routing user queries to the correct tools.

CRITICAL INSTRUCTIONS:
1. TONE: Be polite but brutally honest. Never sugarcoat financial risks.
2. NUMERICAL ACCURACY: ZERO-TOLERANCE for hallucinations. Always rely on tools.
3. LANGUAGE: Use natural English, or switch smoothly into Hinglish if the user chats in Hindi.
4. FORMAT: Use structured bullet points for output answers."""

# ── PERSISTENT MEMORY MANAGEMENT ──────────────────────────
MEMORY_FILE = "tax.memory.json"

def load_memory():
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                return messages_from_dict(json.load(f))
    except Exception as e:
        print(f"Error loading memory: {e}")
    return []

def save_memory(chat_history):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(messages_to_dict(chat_history), f)
    except Exception as e:
        print(f"Memory save error: {e}")

def trim_message(hist, max_message=10):
    sys_msg = [msg for msg in hist if isinstance(msg, SystemMessage)]
    recent_msg = [msg for msg in hist if not isinstance(msg, SystemMessage)][-max_message:]
    return sys_msg + recent_msg

# ── VECTOR STORAGE (RAG) ─────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="savetax_knowledge")

# ── TOOLS ────────────────────────────────────────────────
web_search = DuckDuckGoSearchRun()

@tool
def CAlocator(city: str) -> str:
    """Find corporate tax firms and nearby expert Chartered Accountants based on location."""
    try:
        return web_search.run(f"Chartered Accountant firm office in {city} contact phone details")
    except Exception as e:
        return f"Web search constraint: {e}"

@tool
def invoice(client_name: str, items_json: str, is_inter_state: bool = False) -> str:
    """Generates a professional, beautifully styled HTML invoice bill and saves it locally."""
    try:
        items = json.loads(items_json)
        inv_no = f"INV-{datetime.date.today().strftime('%Y%m%d')}-01"
        date_str = datetime.date.today().strftime("%d-%b-%Y")
        grand_taxable, grand_gst, grand_total = 0.0, 0.0, 0.0
        text_rows = ""
        for item in items:
            desc = item["desc"]
            qty, rate, gst_pct = int(item["qty"]), float(item["rate"]), float(item["gst"])
            taxable = qty * rate
            tax_amount = taxable * (gst_pct / 100)
            total = taxable + tax_amount
            grand_taxable += taxable
            grand_gst += tax_amount
            grand_total += total
            tax_label = f"{gst_pct}% IGST" if is_inter_state else f"{gst_pct}% (CGST+SGST)"
            text_rows += f"| {desc} | {qty} | ₹{rate} | {tax_label} | ₹{tax_amount} | ₹{total} |\n"

        raw_text_invoice = f"""==================================================
                  TAX INVOICE                     
==================================================
Invoice No: {inv_no}          Date: {date_str}
Billed To : {client_name}
--------------------------------------------------
{text_rows}--------------------------------------------------
Taxable Amount : ₹{grand_taxable:,.2f}
Total GST      : ₹{grand_gst:,.2f}
GRAND TOTAL    : ₹{grand_total:,.2f}
==================================================\n"""
        return f"### Invoice Output\n```text\n{raw_text_invoice}\n```"
    except Exception as e:
        return f"Error building invoice layout: {str(e)}"

@tool
def gst_calculator(base_amount: float, gst_rate_pct: float, is_inter_state: bool = False) -> str:
    """Calculates the GST breakdown for a given base amount and tax rate."""
    try:
        base, rate = float(base_amount), float(gst_rate_pct)
        total_gst = base * (rate / 100.0)
        grand_total = base + total_gst
        cgst = sgst = 0.0 if is_inter_state else total_gst / 2.0
        igst = total_gst if is_inter_state else 0.0
        result = {
            "status": "success",
            "calculations": {
                "base_amount": round(base, 2), "gst_rate_percentage": round(rate, 2),
                "cgst": round(cgst, 2), "sgst": round(sgst, 2), "igst": round(igst, 2),
                "total_gst_amount": round(total_gst, 2), "grand_total": round(grand_total, 2)
            }
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@tool
def save_tax(question: str) -> str:
    """Search the RAG knowledge base for Indian tax saving information."""
    try:
        results = collection.query(query_texts=[question], n_results=1)
        docs = results.get("documents", [[]])
        return "\n".join(docs[0]) if docs and docs[0] else "No tax parameters indexed locally."
    except Exception as e:
        return f"RAG error: {e}"

# ── ✅ CACHED AGENT SETUP ─────────────────────────────────
@st.cache_resource
def get_agent():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",  # Or llama-3.3-70b-versatile based on your plan
        temperature=0.6
    )
    tools = [gst_calculator, save_tax, CAlocator, invoice, web_search]
    return create_react_agent(llm, tools)

agent_executor = get_agent()

# ── STATE INITIALIZATION ─────────────────────────────────
if "chat_history" not in st.session_state:
    history = load_memory()
    if not history or not any(isinstance(m, SystemMessage) for m in history):
        history.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    st.session_state.chat_history = history

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# ── SIDEBAR INTERFACE ────────────────────────────────────
with st.sidebar:
    st.markdown("## 💼 PocketCA")
    st.markdown("---")
    st.markdown("**What I can help with:**")
    st.markdown("📊 GST Calculator\n\n💰 Income Tax Planning\n\n🧾 GST Invoice Generation\n\n🏦 Tax Saving Suggestions")
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = [SystemMessage(content=SYSTEM_PROMPT)]
        st.session_state.display_messages = []
        st.session_state.prefill = ""
        save_memory(st.session_state.chat_history)
        st.rerun()

# ── MAIN APPLICATION INTERFACE ───────────────────────────
st.markdown("# 💼 PocketCA")
st.markdown("Your AI Chartered Accountant — GST, Tax Planning, Invoices & Financial Guidance.")
st.markdown("---")

st.markdown("### 💡 Try these:")
col1, col2, col3, col4 = st.columns(4)

if col1.button("📊 Calculate GST"):
    st.session_state.prefill = "Calculate GST for ₹50,000 at 18%"
if col2.button("💰 Save Tax"):
    st.session_state.prefill = "My salary is ₹12 lakh. How can I save maximum tax?"
if col3.button("🧾 Generate Invoice"):
    st.session_state.prefill = "Generate invoice for Rohan: 5 items of consulting work, rate 2000 each, 18% gst"
if col4.button("🏦 Tax Deductions"):
    st.session_state.prefill = "Explain all deductions under 80C, 80D and NPS"

st.markdown("---")

# Render active layout logs using display_messages tracker
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Capture live text entries
user_msg = st.chat_input("Ask PocketCA a financial question...")

if st.session_state.prefill and not user_msg:
    user_msg = st.session_state.prefill
    st.session_state.prefill = ""

if user_msg:
    st.session_state.display_messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)
    
    st.session_state.chat_history.append(HumanMessage(content=user_msg))
    trimmed_context = trim_message(st.session_state.chat_history, max_message=6)
    
    with st.chat_message("assistant"):
        with st.spinner("PocketCA processing metrics..."):
            try:
                response = agent_executor.invoke({"messages": trimmed_context})
                ai_msg = response["messages"][-1]
                
                st.write(ai_msg.content)
                st.session_state.display_messages.append({"role": "assistant", "content": ai_msg.content})
                st.session_state.chat_history.append(ai_msg)
                save_memory(st.session_state.chat_history)
            except Exception as e:
                st.error(f"Execution Error: {e}")
                st.session_state.chat_history.pop()
    st.rerun()
