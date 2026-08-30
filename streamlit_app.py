import streamlit as st
import requests
import os

API_BASE_URL = "http://localhost:8000/api"

# ── Deployment Mode & Auth Config ───────────────────────────────────
DEPLOYMENT_MODE = os.environ.get("DEPLOYMENT_MODE", "dev").lower()
# In production the UI reads the Bearer token from the STREAMLIT_AUTH_TOKEN env var.
# In dev mode this can be left empty and the dropdown selection is used instead.
AUTH_TOKEN = os.environ.get("STREAMLIT_AUTH_TOKEN", "")

def _auth_headers() -> dict:
    """Return Authorization header dict for all backend requests."""
    if AUTH_TOKEN:
        return {"Authorization": f"Bearer {AUTH_TOKEN}"}
    return {}

st.set_page_config(
    page_title="NatWest RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

# ── Dev Mode Banner ──────────────────────────────────────────────
if DEPLOYMENT_MODE == "dev":
    st.warning(
        "⚠️ **DEV MODE — RBAC SIMULATED, NOT ENFORCED.** "
        "Role/department dropdowns below are for local testing only. "
        "This banner must never appear in a production deployment.",
        icon="🚨"
    )

# Initialize Session State
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
# Cache the server-resolved identity (production mode)
if "server_identity" not in st.session_state:
    st.session_state.server_identity = None

def load_stats():
    try:
        res = requests.get(f"{API_BASE_URL}/stats", headers=_auth_headers())
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"total_documents": 0, "total_chunks": 0, "total_chats": 0, "total_tokens_saved": 0}

def load_documents():
    try:
        res = requests.get(f"{API_BASE_URL}/documents", headers=_auth_headers())
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

def load_chats():
    try:
        res = requests.get(f"{API_BASE_URL}/chats", headers=_auth_headers())
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

# Sidebar
with st.sidebar:
    st.title("🧠 NatWest RAG Assistant")

    # ── User Profile ─────────────────────────────────────────────
    if DEPLOYMENT_MODE == "production":
        # ── Production: fetch server-resolved identity from /api/me
        if st.session_state.server_identity is None:
            try:
                me_res = requests.get(
                    f"{API_BASE_URL}/me", headers=_auth_headers(), timeout=3
                )
                if me_res.status_code == 200:
                    st.session_state.server_identity = me_res.json()
            except Exception:
                pass

        identity = st.session_state.server_identity or {}
        sim_role = identity.get("role", "Unknown")
        sim_dept = identity.get("department", "Unknown")

        st.header("User Profile")
        st.info(
            f"👤 **Role:** {sim_role}  \n"
            f"🏢 **Department:** {sim_dept}  \n"
            f"*Identity verified via Bearer token.*"
        )
    else:
        # ── Dev mode: keep simulated dropdowns
        st.header("Simulate User Profile")
        sim_role = st.selectbox(
            "User Role",
            ["Teller", "Manager", "Executive", "Admin"],
            index=0,
            help="DEV ONLY — In production, role is derived from the Bearer token."
        )
        sim_dept = st.selectbox(
            "User Department",
            ["Retail", "Lending", "Compliance", "Wealth"],
            index=0
        )

    st.divider()
    
    # ── Upload Document Section with Banking Metadata ───────────────
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Drop document here", type=["pdf", "xlsx", "xls", "docx", "doc"])
    
    if uploaded_file is not None:
        file_sig = uploaded_file.name
        
        # Metadata inputs
        st.subheader("Document Metadata")
        clearance = st.selectbox("Clearance Level", ["Public", "Internal", "Restricted"], index=1)
        doc_dept = st.selectbox("Document Department", ["Retail", "Lending", "Compliance", "Wealth"], index=0)
        
        import datetime
        eff_date = st.date_input("Effective Date", datetime.date.today())
        exp_date = st.date_input("Expiry Date", datetime.date(2099, 12, 31))

        if file_sig not in st.session_state.processed_files:
            if st.button("🚀 Process & Index Document"):
                with st.spinner("Uploading file..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    params = {
                        "clearance_level": clearance,
                        "department": doc_dept,
                        "effective_date": eff_date.isoformat(),
                        "expiry_date": exp_date.isoformat()
                    }
                    try:
                        res = requests.post(
                            f"{API_BASE_URL}/upload",
                            files=files,
                            params=params,
                            headers=_auth_headers()
                        )
                        if res.status_code == 200:
                            st.success(f"Uploaded {uploaded_file.name}! Processing in background.")
                            st.session_state.processed_files.add(file_sig)
                            st.rerun()
                        else:
                            try:
                                err = res.json().get('detail', 'Upload failed')
                            except Exception:
                                err = f"Status {res.status_code}: {res.text}"
                            st.error(f"Backend Error: {err}")
                    except Exception as e:
                        st.error(f"Failed to connect to backend: {e}")
        else:
            st.success(f"✓ {uploaded_file.name} is uploaded!")

    st.divider()
    
    st.header("Documents")
    docs = load_documents()
    if not docs:
        st.write("No documents yet")
    else:
        for doc in docs:
            status = doc.get("status", "processed")
            if status == "processing":
                st.text(f"🔄 {doc['filename']} (Indexing...)")
            elif status == "failed":
                st.text(f"❌ {doc['filename']} (Failed)")
            else:
                st.text(f"✅ {doc['filename']} ({doc['chunk_count']} chunks)")

    st.divider()

    st.header("Chats")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("New Chat"):
            st.session_state.chat_id = None
            st.session_state.messages = []
            st.rerun()

    chats = load_chats()
    if not chats:
        st.write("No chats yet")
    else:
        for chat in chats:
            chat_id = chat['chat_id']
            short_id = chat_id[:8] + "..."
            if st.button(f"💬 {short_id} ({chat['turn_count']} turns)", key=chat_id):
                st.session_state.chat_id = chat_id
                st.session_state.messages = []  # Clear UI history since backend doesn't store full logs
                st.rerun()

    st.divider()
    
    st.header("Token Savings")
    stats = load_stats()
    st.metric(label="Total Tokens Saved", value=stats.get("total_tokens_saved", 0))
    st.write(f"**Documents**: {stats.get('total_documents', 0)}")
    st.write(f"**Chunks**: {stats.get('total_chunks', 0)}")
    st.write(f"**Chats**: {stats.get('total_chats', 0)}")


# Main Chat Area
st.title("RAG Assistant Chat")

if not docs:
    st.info("Upload a document in the sidebar to get started!")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            payload = {
                "question": prompt,
                "user_role": sim_role,
                "user_department": sim_dept
            }
            if st.session_state.chat_id:
                payload["chat_id"] = st.session_state.chat_id
                
            res = requests.post(
                f"{API_BASE_URL}/ask",
                json=payload,
                headers=_auth_headers(),
                stream=True
            )
            
            if res.status_code == 200:
                import json
                
                def stream_parser(response):
                    sources_to_append = ""
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                                try:
                                    data = json.loads(data_str)
                                    if data["type"] == "metadata":
                                        st.session_state.chat_id = data["chat_id"]
                                        sources = data.get("sources", [])
                                        if sources:
                                            sources_to_append = "\n\n**Sources:**\n"
                                            for i, source in enumerate(sources, 1):
                                                sources_to_append += f"- {source['document_name']} (Page {source.get('page_number', 'N/A')})\n"
                                    elif data["type"] == "chunk":
                                        yield data["content"]
                                except Exception as e:
                                    pass
                    if sources_to_append:
                        yield sources_to_append

                # Streamlit magically animates any generator passed to write_stream
                message_placeholder.empty() # Clear the "Thinking..." text
                response_text = st.write_stream(stream_parser(res))
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                try:
                    error_msg = res.json().get("detail", "Error processing request")
                except:
                    error_msg = f"HTTP {res.status_code}"
                message_placeholder.markdown(f"**Error:** {error_msg}")
                st.session_state.messages.append({"role": "assistant", "content": f"**Error:** {error_msg}"})
        except Exception as e:
            message_placeholder.markdown(f"**Failed to connect to backend:** {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"**Failed to connect to backend:** {e}"})
