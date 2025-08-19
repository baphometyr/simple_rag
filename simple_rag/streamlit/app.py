from simple_rag.index import IndexBuilder, Embedding
from simple_rag.llm import OpenAIProvider, AzureOpenAIProvider, LocalModelProvider
from simple_rag.pipeline import Pipeline

import streamlit as st
import PyPDF2

ASSISTANT_MESSAGE = "Hello! I'm here to answer your questions."


# --- ONCHANGE FUNCIONS ---
def on_collection_change():
    if st.session_state.collection:
        embedding = Embedding()
        st.session_state.index = IndexBuilder.load_collection(
            st.session_state.collection, 
            embedding)

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
        embedding_model = st.text_input("Embedding model", "all-MiniLM-L6-v2", disabled=True )
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
            
@st.dialog("Configure OpenAI parameters:")
def configure_openai():
    api_key = st.text_input("API Key", type="password")
    model_name = st.text_input("Model")

    if st.button("Save"):
        try:
            st.session_state.model = OpenAIProvider(
                api_key=api_key,
                model_name=model_name)
            st.success("Model configured successfully!")

        except:
            st.error("Error configuring OpenAI. Please check your parameters.")


@st.dialog("Configure AzureOpenAI parameters:")
def configure_azureopenai():
    api_key = st.text_input("API Key", type="password")
    azure_endpoint = st.text_input("Azure Endpoint")
    api_version = st.text_input("API Version")
    deployment_name = st.text_input("Deployment Name")

    if st.button("Save"):
        try:
            st.session_state.model = AzureOpenAIProvider(
                api_key = api_key,
                azure_endpoint = azure_endpoint,
                api_version = api_version,
                deployment_name = deployment_name)
            st.success("Model configured successfully!")

        except Exception as e:
            st.error(e)
            st.error("Error configuring AzureOpenAI. Please check your parameters.")



@st.dialog("Configure Local model parameters:")
def configure_local():
    model_name = st.text_input("Model Name (Hugging Face)")
    hf_token = st.text_input("Hugging Face Token (if necessary)", type="password")
    
    if st.button("Save"):
        try:
            with st.spinner("Downloading and Load Local Model..."):
                st.session_state.model = LocalModelProvider(
                    model_name=model_name,
                    hf_token=hf_token)
                st.success("Model configured successfully!")

        except:
            st.error("Error configuring Local model. Please check your parameters.")


# --- SIDE BAR ---
with st.sidebar:
    st.header("Settings")

    # Collection
    st.selectbox(
        "Collection:",
        IndexBuilder.list_collections(),
        key='collection',
        index=None,
        on_change=on_collection_change,
        placeholder="Select a collection...", 
        label_visibility="collapsed")
    
    if st.button("Create collection"):
        build_collection()

    # LLM
    st.selectbox(
        "LLM:",
        ["OpenAI", "Azure OpenAI", "Local (HuggingFace)"],
        key='llm',
        index=None,
        placeholder="Select a LLM...", 
        label_visibility="collapsed")
    
    if st.session_state.llm == "OpenAI":
        configure_openai()
    elif st.session_state.llm == "Azure OpenAI":
        configure_azureopenai()
    elif st.session_state.llm == "Local (HuggingFace)":
        configure_local()



    

# --- MESSAGES ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": ASSISTANT_MESSAGE}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- CHAT ---
if user_query := st.chat_input():
    if 'index' not in st.session_state:
        st.error("Please create or load a collection first.")
        st.stop()

    if 'model' not in st.session_state:
        st.error("Please configure the model first.")
        st.stop()
    

    # write user query and save to session state
    st.chat_message("user").write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # model response
    pipeline = Pipeline(
        index=st.session_state.index,
        llm=st.session_state.model,
        )

    model_response = pipeline.run(user_query)

    # write assistant response and save to session state
    st.chat_message("assistant").write(model_response)
    st.session_state.messages.append({"role": "assistant", "content": model_response})