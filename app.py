import json
import datetime
import requests
import os
import streamlit as st
import chromadb
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (
    messages_from_dict,
    messages_to_dict,
    SystemMessage,
    HumanMessage,
    AIMessage
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ── STREAMLIT PAGE CONFIG ────────────────────────────────
st.set_page_config(
    page_title="PocketCA",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: white; }
    [data-testid="stSidebar"] { background-color: #1e293b; }
    h1 { color: #38bdf8; }
    .stButton > button {
        background-color: #1e40af;
        color: white;
        border-radius: 10px;
        border: none;
        width: 100%;
        padding: 12px;
        font-size: 14px;
    }
    .stButton > button:hover { background-color: #2563eb; color: white; }
</style>
""", unsafe_allow_html=True)

# Direct configuration without if/else blocks
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ── Persistent memory ────────────────────────────────────
MEMORY_FILE = "tax.memory.json"

def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return messages_from_dict(json.load(f))
    except:
        return []

def save_memory(chat_history):
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(messages_to_dict(chat_history), f)
    except Exception as e:
        print(f"Memory save error: {e}")

def trim_message(hist, max_message=10):
    if len(hist) <= max_message:
        return hist
    sys_msg = [msg for msg in hist if isinstance(msg, SystemMessage)]
    recent_msg = [msg for msg in hist if not isinstance(msg, SystemMessage)][-max_message:]
    return sys_msg + recent_msg

# ── RAG: ChromaDB ────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="savetax_knowledge")

TAX_KNOWLEDGE = """🛡️ Section 80C (Limit: ₹1,50,000/year):
EPF, PPF, ELSS, NSC, SSY, NPS, Life Insurance Premium,
Children Tuition Fees, Home Loan Principal, Tax-Saving FDs.

🏥 Section 80D (Medical Insurance):
Self & Family: up to ₹25,000. Parents under 60: ₹25,000.
Senior citizen parents: ₹50,000. Preventive checkup: ₹5,000.

🏠 Home & Education Loans:
Section 24(b): up to ₹2,00,000 on home loan interest.
Section 80E: full interest on education loan, no limit, 8 years.

👴 Additional Deductions:
80CCD(1B): extra ₹50,000 for NPS.
80TTA: ₹10,000 on savings account interest (₹50,000 for seniors under 80TTB).
80G: 50-100% on donations to registered NGOs.
HRA / 80GG: up to ₹60,000/year for non-salaried rent payers."""

if collection.count() == 0:
    collection.add(
        documents=[TAX_KNOWLEDGE],
        ids=["0"]
    )

# ── Tools ─────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date general knowledge, current events, and financial tax updates."""
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        return DuckDuckGoSearchAPIWrapper().run(query)
    except Exception as e:
        return f"Search engine exception: {e}"

@tool
def CAlocator(city: str) -> str:
    """If the user has a complex tax notice or penalty, suggest a CA nearby.
    Use DuckDuckGoSearchRun immediately to find accurate CA details:
    name, qualification, fees, location."""
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        return DuckDuckGoSearchAPIWrapper().run(f"Chartered Accountant firm office in {city} contact phone details")
    except Exception as e:
        return f"Web search constraint: {e}"

@tool
def invoice(client_name: str, items_json: str, is_inter_state: bool = False) -> str:
    """Generate a professional GST-compliant invoice from the user's description.
    Include: invoice number, date, vendor/client details, line items,
    HSN/SAC codes, tax breakdown (CGST/SGST/IGST), and grand total.
    If HSN codes or tax rates are missing, search the web first."""
    try:
        items = json.loads(items_json)
        inv_no = f"INV-{datetime.date.today().strftime('%Y%m%d')}-01"
        date_str = datetime.date.today().strftime("%d-%b-%Y")
        grand_taxable, grand_gst, grand_total = 0.0, 0.0, 0.0
        text_rows = ""
        for item in items:
            desc = item.get("desc", item.get("item_name", "Item"))
            qty = int(item.get("qty", item.get("quantity", 1)))
            rate = float(item.get("rate", item.get("price", 0.0)))
            gst_pct = float(item.get("gst", item.get("gst_rate", 18.0)))
            
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
    """Calculates the GST breakdown for a given base amount and tax rate.
    - base_amount: taxable value
    - gst_rate_pct: GST rate (e.g. 5, 12, 18, 28)
    - is_inter_state: True = IGST applies; False = CGST + SGST split"""
    try:
        base = float(base_amount)
        rate = float(gst_rate_pct)
        total_gst = base * (rate / 100.0)
        grand_total = base + total_gst

        if is_inter_state:
            cgst, sgst, igst = 0.0, 0.0, total_gst
        else:
            cgst = sgst = total_gst / 2.0
            igst = 0.0

        result = {
            "status": "success",
            "calculations": {
                "base_amount": round(base, 2),
                "gst_rate_percentage": round(rate, 2),
                "cgst": round(cgst, 2),
                "sgst": round(sgst, 2),
                "igst": round(igst, 2),
                "total_gst_amount": round(total_gst, 2),
                "grand_total": round(grand_total, 2)
            }
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@tool
def save_tax(question: str) -> str:
    """Search the RAG knowledge base for Indian tax saving information.
    If not found, use DuckDuckGo to get the correct answer."""
    try:
        results = collection.query(query_texts=[question], n_results=2)
        docs = results.get("documents", [[]])
        if docs and docs[0]:
            return "\n".join(docs[0])
        return "No relevant info found in knowledge base. Please search the web."
    except Exception as e:
        return f"RAG error: {e}"

# ── LLM AND TOOL CALLING ─────────────────────────────────
@st.cache_resource
def get_agent():
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.6)
    tools = [gst_calculator, save_tax, CAlocator, invoice, web_search]
    return create_react_agent(llm, tools)

agent_executor = get_agent()

# ── System prompt ─────────────────────────────────────────
SYSTEM_PROMPT = """You are the Pocket CA Agent, a highly precise, reliable, and brutally honest AI financial assistant.
Your primary function is to act as an expert Chartered Accountant, routing user queries to the correct tools.

CRITICAL INSTRUCTIONS:

1. TONE: Be polite but brutally honest. Never sugarcoat financial risks or tax liabilities.
   If the user is losing money or calculating something wrong, state the truth directly.

2. NUMERICAL ACCURACY: ZERO-TOLERANCE for hallucinations on numbers.
   Always use your tools (gst_calculator, save_tax) for math. Never guess or approximate.

3. LANGUAGE: Use natural english — a casual, professional mix of English and Hindi.
   Avoid heavy Sanskritized Hindi or overly formal English.
   default language use english and if user speaks in hindi then contiune in hindi

4. FORMAT: Always use structured bullet points for answers.

EXAMPLE:
User: "Mera income 12 Lakhs hai, kitna tax hoga?"
Response:
- Standard Deduction: ₹75,000 deducted.
- Taxable Income: ₹11,25,000 remaining.
- Tax Liability (New Regime): ₹X,XXX calculated.
- Honest Review: Missing 80C/80D deductions — you may be losing money."""

# ── STATE INITIALIZATION ─────────────────────────────────
if "chat_history" not in st.session_state:
    history = load_memory()
    if not any(isinstance(m, SystemMessage) for m in history):
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
    st.markdown("📊 GST Calculator")
    st.markdown("💰 Income Tax Planning")
    st.markdown("🧾 GST Invoice Generation")
    st.markdown("🏦 Tax Saving Suggestions")
    st.markdown("👨‍💼 CA Assistance")
    st.markdown("📑 Tax Deductions (80C, 80D, NPS)")
    st.markdown("🔍 Tax & GST Information")
    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = [SystemMessage(content=SYSTEM_PROMPT)]
        st.session_state.display_messages = []
        st.session_state.prefill = ""
        save_memory(st.session_state.chat_history)
        st.rerun()
    st.markdown("---")
    st.caption("Built with LangGraph + Groq + Streamlit + LLM + Memory")

# ── MAIN APPLICATION INTERFACE ───────────────────────────
st.markdown("# 💼 PocketCA")
st.markdown("Your AI Chartered Accountant — GST, Tax Planning, Invoices & Financial Guidance.")
st.markdown("---")

st.markdown("### 💡 Try these:")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Calculate GST"):
        st.session_state.prefill = "Calculate GST for ₹50,000 at 18%"
        st.rerun()
with col2:
    if st.button("💰 Save Tax"):
        st.session_state.prefill = "My salary is ₹12 lakh. How can I save maximum tax?"
        st.rerun()
with col3:
    if st.button("🧾 Generate Invoice"):
        st.session_state.prefill = "Generate GST invoice for software development service worth ₹25,000"
        st.rerun()
with col4:
    if st.button("🏦 Tax Deductions"):
        st.session_state.prefill = "Explain all deductions under 80C, 80D and NPS"
        st.rerun()

st.markdown("---")

# Render historical context
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Capture text entries
user_msg = st.chat_input("Ask PocketCA a financial question...")

if st.session_state.prefill and not user_msg:
    user_msg = st.session_state.prefill
    st.session_state.prefill = ""

if user_msg:
    st.session_state.display_messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)
    
    st.session_state.chat_history.append(HumanMessage(content=user_msg))
    trimmed_context = trim_message(st.session_state.chat_history, max_message=5)
    
    with st.chat_message("assistant"):
        with st.spinner("PocketCA processing metrics..."):
            try:
                response = agent_executor.invoke({"messages": trimmed_context})
                st.session_state.chat_history = response["messages"]
                ai_msg = st.session_state.chat_history[-1]
                
                # Filter out raw function calling background metadata blocks
                clean_content = ai_msg.content
                if "</function>" in clean_content:
                    clean_content = clean_content.split("</function>")[-1].strip()
                elif "function>" in clean_content:
                    clean_content = clean_content.split("function>")[-1].strip()
                
                st.write(clean_content)
                st.session_state.display_messages.append({"role": "assistant", "content": clean_content})
                save_memory(st.session_state.chat_history)
            except Exception as e:
                st.error(f"Error occurred: {e}")
                if st.session_state.chat_history:
                    st.session_state.chat_history.pop()
    st.rerun()
