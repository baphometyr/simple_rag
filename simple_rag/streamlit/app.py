from simple_rag.index import IndexBuilder, Embedding
from simple_rag.llm import OpenAIProvider, AzureOpenAIProvider, LocalModelProvider
from simple_rag.pipeline import Pipeline

import streamlit as st
import PyPDF2
import csv
import io

ASSISTANT_MESSAGE = "Hello! I'm here to answer your questions."


# --- INIT VALUES ---
if 'azure_api_key' not in st.session_state:
    st.session_state.azure_api_key = ""
if 'azure_endpoint' not in st.session_state:
    st.session_state.azure_endpoint = ""
if 'azure_api_version' not in st.session_state:
    st.session_state.azure_api_version = ""
if 'azure_deployment_name' not in st.session_state:
    st.session_state.azure_deployment_name = ""

if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'openai_model_name' not in st.session_state:
    st.session_state.openai_model_name = ""

if 'local_hf_token' not in st.session_state:
    st.session_state.local_hf_token = ""
if 'local_model_name' not in st.session_state:
    st.session_state.local_model_name = ""

# --- ONCHANGE FUNCIONS ---
def on_collection_change():
    if st.session_state.collection:
        embedding = Embedding()
        st.session_state.index = IndexBuilder.load_collection(
            st.session_state.collection, 
            embedding)

def on_llm_change():
    if st.session_state.choice_llm == "OpenAI":
        configure_openai()
    elif st.session_state.choice_llm == "Azure OpenAI":
        configure_azureopenai()
    elif st.session_state.choice_llm == "Local (HuggingFace)":
        configure_local()

# --- MODALS ---
@st.dialog("Create your collection:", dismissible=False)
def build_collection():
    # collection input
    texts = []
    collection_name = st.text_input("Collection name", max_chars=50)
    collection_path = st.text_input("Collection path", "indices/", max_chars=100)

    # Extension file
    extension = st.selectbox(
        "Extension File",
        ("PDF", "CSV"),
        index=None,
        placeholder="Select extension file to upload...",
    )

    if extension == 'PDF':
        # Upload pdf files
        uploaded_files = st.file_uploader("Choose a PDF files", accept_multiple_files=True)
        for uploaded_file in uploaded_files:                
            if uploaded_file.type != 'application/pdf':
                st.error('Some files are not PDFs', icon="🚨")
            else:
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            texts.append(text)
                except:
                    st.error(f"Error reading file {uploaded_file.name}, please check the contents or delete it.")

        # Chunking type
        opcions = ['Page', 'Recursive', 'Character']
        st.segmented_control("Chunking Type", opcions, disabled=True)

    if extension == 'CSV':
        # Upload pdf files
        col_name = st.text_input("Column name to extract")
        uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False, disabled = col_name=="")
        
        if uploaded_file:
            if uploaded_file.type != 'text/csv':
                st.error('File is not CSV', icon="🚨")
            else:
                # extract texts
                try:
                    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                    reader = csv.DictReader(stringio)
                    texts = [row[col_name] for row in reader]
                except:
                    st.error("Error reading CSV file, Please verify that the column exists in the file.")
        
        opcions = ['None', 'Recursive', 'Character']
        st.segmented_control("Chunking Type", opcions, disabled=True)
        

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
    st.session_state.openai_api_key = st.text_input("API Key", value=st.session_state.openai_api_key, type="password")
    st.session_state.openai_model_name = st.text_input("Model", value=st.session_state.openai_model_name)

    if st.button("Save"):
        try:
            st.session_state.model = OpenAIProvider(
                api_key=st.session_state.openai_api_key,
                model_name=st.session_state.openai_model_name)
            st.success("Model configured successfully!")

        except:
            st.error("Error configuring OpenAI. Please check your parameters.")


@st.dialog("Configure AzureOpenAI parameters:")
def configure_azureopenai():
    st.session_state.azure_api_key = st.text_input("API Key", value=st.session_state.azure_api_key, type="password")
    st.session_state.azure_endpoint = st.text_input("Azure Endpoint", value=st.session_state.azure_endpoint)
    st.session_state.azure_api_version = st.text_input("API Version", value=st.session_state.azure_api_version)
    st.session_state.azure_deployment_name = st.text_input("Deployment Name", value=st.session_state.azure_deployment_name)

    if st.button("Save"):
        try:
            st.session_state.model = AzureOpenAIProvider(
                api_key = st.session_state.azure_api_key,
                azure_endpoint = st.session_state.azure_endpoint,
                api_version = st.session_state.azure_api_version,
                deployment_name = st.session_state.azure_deployment_name)
            st.success("Model configured successfully!")

        except Exception as e:
            st.error(e)
            st.error("Error configuring AzureOpenAI. Please check your parameters.")



@st.dialog("Configure Local model parameters:")
def configure_local():
    st.info('This model is loaded locally, performance depends entirely on your computing resources.')

    st.session_state.local_model_name = st.text_input("Model Name (Hugging Face)", value=st.session_state.local_model_name)
    st.session_state.local_hf_token = st.text_input("Hugging Face Token (if necessary)", value=st.session_state.local_hf_token, type="password")
    
    if st.button("Save"):
        try:
            with st.spinner("Downloading and Load Local Model..."):
                st.session_state.model = LocalModelProvider(
                    model_name=st.session_state.local_model_name,
                    hf_token=st.session_state.local_hf_token)
                st.success("Model configured successfully!")
        except:
            st.error("Error configuring Local model. Please check your parameters.")
    st.session_state.choice_llm = ""


# --- SIDE BAR ---
with st.sidebar:
    # Collection
    st.header("Collection:")

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
    st.header("LLM:")
    st.selectbox(
        "LLM:",
        ["OpenAI", "Azure OpenAI", "Local (HuggingFace)"],
        index=None,
        key='choice_llm',
        on_change=on_llm_change,
        placeholder="Select a LLM...", 
        label_visibility="collapsed")
    

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