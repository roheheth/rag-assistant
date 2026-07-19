import streamlit as st
import requests
import os

API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize Session State
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

def load_stats():
    try:
        res = requests.get(f"{API_BASE_URL}/stats")
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"total_documents": 0, "total_chunks": 0, "total_chats": 0, "total_tokens_saved": 0}

def load_documents():
    try:
        res = requests.get(f"{API_BASE_URL}/documents")
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

def load_chats():
    try:
        res = requests.get(f"{API_BASE_URL}/chats")
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

# Sidebar
with st.sidebar:
    st.title("🧠 RAG Assistant")
    
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Drop document here", type=["pdf", "xlsx", "xls", "docx", "doc"])
    if uploaded_file is not None:
        file_sig = uploaded_file.name
        if file_sig not in st.session_state.processed_files:
            with st.spinner("Uploading file..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    res = requests.post(f"{API_BASE_URL}/upload", files=files)
                    if res.status_code == 200:
                        st.success(f"Uploaded {uploaded_file.name}! Processing in background.")
                        st.session_state.processed_files.add(file_sig)
                        st.rerun()  # Rerun to unlock chat screen immediately
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
            payload = {"question": prompt}
            if st.session_state.chat_id:
                payload["chat_id"] = st.session_state.chat_id
                
            res = requests.post(f"{API_BASE_URL}/ask", json=payload, stream=True)
            
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
