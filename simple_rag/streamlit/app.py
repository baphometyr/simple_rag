from simple_rag.index import IndexBuilder, Embedding

import streamlit as st
import PyPDF2

ASSISTANT_MESSAGE = "Hello! I'm here to answer your questions."


# --- ONCHANGE FUNCIONS ---

# --- MODALS ---
@st.dialog("Create your collection:", dismissible=False)
def build_collection():
    # collection input
    texts = []
    collection_name = st.text_input("Collection name", max_chars=50)
    collection_path = st.text_input("Collection path", "indices/", max_chars=100)

    # Upload pdf files
    uploaded_files = st.file_uploader("Choose a PDF files", accept_multiple_files=True)
    for uploaded_file in uploaded_files:                
        if uploaded_file.type != 'application/pdf':
            st.error('Some files are not PDFs', icon="🚨")
        else:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)

    # Advanced options
    with st.expander("Advanced options"):
        embedding_model = st.text_input("Embedding model", "all-MiniLM-L6-v2")
        index_type = st.selectbox("Type of index:", ("flat", "ivf"))
        clusters = st.number_input("Number of clusters:", 10, 1000, 10, step=10)

            
    col1, col2, _ = st.columns([1, 1, 1])
    # button create collection
    with col1:
        if st.button("Create collection"):
            if collection_name and collection_path and texts:
                try:
                    # build embedding
                    embedding = Embedding(model_name=embedding_model)

                    # build collection
                    index = IndexBuilder(embedding=embedding, base_dir=collection_path)
                    index.build_collection(
                        collection_name, 
                        texts, 
                        use_advanced_index=(index_type == "ivf"),
                        nlist=clusters
                        )
                    st.session_state.index = index
                    st.session_state.collection = collection_name

                
                except Exception as e:
                    st.error(e)
                else:
                    st.success("Collection created successfully!")
                    st.rerun()
            
            st.error("Please fill in all fields.")

    # button cancel collection
    with col2:
        if st.button("Cancel"):
            st.rerun()
            

# --- SIDE BAR ---
with st.sidebar:
    collection = st.selectbox(
        "Collection:",
        IndexBuilder.list_collections(),
        key='collection',
        index=None, 
        placeholder="Select a collection...", 
        label_visibility="collapsed")
    
    if st.button("Create collection"):
        build_collection()

    


# --- MESSAGES ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": ASSISTANT_MESSAGE}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- CHAT ---
if user_query := st.chat_input():
    # write user query and save to session state
    st.chat_message("user").write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # model response
    model_response = "test response"
    st.write(st.session_state.index.texts)

    # write assistant response and save to session state
    st.chat_message("assistant").write(model_response)
    st.session_state.messages.append({"role": "assistant", "content": model_response})


