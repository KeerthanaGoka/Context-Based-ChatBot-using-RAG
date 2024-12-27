import streamlit as st
from streamlit_float import *

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document Loaders
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import WikipediaLoader

# Text Splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter

# Chroma DB
from langchain_chroma import Chroma

import os
import re
import shutil
import pickle

from dotenv import load_dotenv

'''
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()
persistent_client = chromadb.PersistentClient()
'''

# Handling Directory Operations
def make_empty_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if item_path.endswith('chat_history.dump') == False:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        print(f"All contents of '{path}' have been deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")

def count_files(directory):
    print("Directory:", directory)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return 0  # Return 0 if directory doesn't exist
    for f in os.listdir(directory):
        print("Files in directory:", f)
    return len([f for f in os.listdir(directory)])

# All document loaders
def getTextDocuments(filename):
    return TextLoader(filename).load()

def getPdfDocuments(filename):
    return PyPDFLoader(filename).load()

def getArxivDocuments(doc_code, max_docs):
    return ArxivLoader(query=doc_code, load_max_docs=max_docs).load()

def getWikiDocuments(query, max_docs):
    return WikipediaLoader(query=query, load_max_docs=max_docs).load()

# All Text Splitters

def getTextSplit(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def getTextCharSplit(docs, chunk_size, chunk_overlap):
    splitter=CharacterTextSplitter(separator="\n\n",chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# Prepare and Get ChromaDB
def getChromsDB(documents, embeddings, path_vdb):
    if not os.path.exists(path_vdb):
        os.makedirs(path_vdb)  # Create the directory if it doesn't exist
    fileCountInDir = count_files(path_vdb)
    print(path_vdb, ":", fileCountInDir)
    if fileCountInDir <= 1:
        vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=path_vdb)
    else:
        print('Using already stored vectorDB')
        vectordb = Chroma(persist_directory=path_vdb, embedding_function=embeddings)
    return vectordb

class gblVariables:
    prev_selected_option    = None
    prev_uploaded_files     = []
    embeddings              = None
    llm                     = None
    retriever               = None

# Main Function
def main():
    session_id              = st.session_state['sessionID']
    user_id                 = st.session_state['userID']
    path_folder_vdb         = '/Users/kapple/Documents/Masters_Study_2/NLP/Project/RAG/vectorDBStore/'
    path_folder_datasource  = '/Users/kapple/Documents/Masters_Study_2/NLP/Project/RAG/datasource/'
    path_folder_users       = '/Users/kapple/Documents/Masters_Study_2/NLP/Project/RAG/users/'
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)
    project_title           = 'Kanji Chatbot'

    print("session_id", session_id)
    if st.session_state['sessionFromChatPage'] == False:
        st.session_state['sessionFromChatPage']   = True
        
        # To be initialized only Once
        load_dotenv()
        gorq_api_key                = os.getenv("GORQ_API_KEY")
        os.environ['HF_TOKEN']      = os.getenv("HF_TOKEN")

        gblVariables.embeddings     = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        gblVariables.llm            = ChatGroq(groq_api_key=gorq_api_key, model_name="Gemma2-9b-It")

    print("::gblVariables.prev_selected_option::", gblVariables.prev_selected_option)
    print("::gblVariables.prev_uploaded_files::", gblVariables.prev_uploaded_files)
    print("::gblVariables.embeddings::", gblVariables.embeddings)
    print("::gblVariables.llm::", gblVariables.llm)

    class Config:
        arbitrary_types_allowed = True

    if 'store' not in st.session_state:
        st.session_state.store  = {}
    print("st.session_state.store", st.session_state.store)

    left, right                 = st.columns([1,2])

    #### Left
    with left:
        with st.container(height=600, border=True, key="left"):

            col1, col2                      = st.columns([4,1], vertical_alignment='bottom')
            col1.title(project_title)
            if session_id.startswith('default_'):
                if col2.button("Login"):
                    st.session_state.page   = 0
                    st.rerun()
            else:
                if col2.button("Logout"):

                    if session_id.startswith('default_') == False:
                        folder_name         = path_folder_users + session_id + '/'
                        count_files(folder_name)
                        if session_id in st.session_state.store:
                            file_name           = re.sub(r'[^a-zA-Z0-9]', '_', gblVariables.prev_selected_option).lower() + '.pkl'
                            print(".................", folder_name + file_name)
                            textToPrint         = st.session_state.store[session_id]
                            print("textToPrint", textToPrint)
                            with open(folder_name + file_name, 'wb') as file:
                                pickle.dump(textToPrint, file)
                    st.session_state.page   = 0
                    st.rerun()
            st.markdown("<i><b>Insights at Your Fingertips - Just Upload and Ask</b></i><br/>", unsafe_allow_html=True)

            userIDContainer             = st.container()
            userIDContainer.markdown('User: ' + user_id)
#            userIDContainer.float("top:500;background-color: white;")


            options                     = ["Clark University International Admissions", 
                                           "Natural Language Processing", 
                                           "Attention is all you need", 
                                           "Upload Your File (PDF)"]
            selected_option             = st.selectbox("Choose an option:", options)
            
            print("::selected_option::Current:", selected_option, ":Prev:", gblVariables.prev_selected_option)
            
            if selected_option == 'Upload Your File (PDF)':
                uploaded_files          = st.file_uploader("Choose A PDf file", type="pdf", accept_multiple_files=True)
                # Custom PDF's
                print("P uploaded_files", uploaded_files)
                print("::uploaded_files::Current:", uploaded_files, ":Prev:", gblVariables.prev_uploaded_files)

                if (uploaded_files != []) and (gblVariables.prev_uploaded_files != uploaded_files):
                    
                    if session_id.startswith('default_') == False:
                        folder_name         = path_folder_users + session_id + '/'
                        count_files(folder_name)
                        if session_id in st.session_state.store:
                            file_name           = re.sub(r'[^a-zA-Z0-9]', '_', gblVariables.prev_selected_option).lower() + '.pkl'
                            print(".................", folder_name + file_name)
                            textToPrint         = st.session_state.store[session_id]
                            print("textToPrint", textToPrint)
                            with open(folder_name + file_name, 'wb') as file:
                                pickle.dump(textToPrint, file)
                    
                    st.session_state.store[session_id]      = ChatMessageHistory()

                    print("Uploaded files Revised........")
                    path_vdb   = path_folder_vdb + session_id + '/'
                    make_empty_directory(path_vdb)
                    documents           = []
                    
                    for uploaded_file in uploaded_files:
                        temppdf         = f"./temp.pdf"
                        with open(temppdf, "wb") as file:
                            file.write(uploaded_file.getvalue())
                        loader          = PyPDFLoader(temppdf)

                        # docs            = getPdfDocuments(path_folder_datasource + 'Empty.pdf')
                        # documents.extend(docs)
                        docs            = loader.load()
                        documents.extend(docs)
                    chunks          = getTextSplit(documents, 4096, 256)
                    vectorstore     = getChromsDB(chunks, gblVariables.embeddings, path_vdb)
                    gblVariables.retriever       = vectorstore.as_retriever()
                    gblVariables.prev_uploaded_files    = uploaded_files
            else: # Some other option
                if gblVariables.prev_selected_option   != selected_option:
                    if session_id.startswith('default_') == False:
                        folder_name         = path_folder_users + session_id + '/'
                        count_files(folder_name)
                        if session_id in st.session_state.store:
                            file_name           = re.sub(r'[^a-zA-Z0-9]', '_', gblVariables.prev_selected_option).lower() + '.pkl'
                            print(".................", folder_name + file_name)
                            textToPrint         = st.session_state.store[session_id]
                            print("textToPrint", textToPrint)
                            with open(folder_name + file_name, 'wb') as file:
                                pickle.dump(textToPrint, file)
                        try:
                            file_name           = re.sub(r'[^a-zA-Z0-9]', '_', selected_option).lower() + '.pkl'
                            with open(folder_name + file_name, "rb") as file:
                                st.session_state.store[session_id]  = pickle.load(file)
                        except:
                            st.session_state.store[session_id]      = ChatMessageHistory()
                    else:
                        st.session_state.store[session_id]      = ChatMessageHistory()
                    
                    folder_name         = re.sub(r'[^a-zA-Z0-9]', '_', selected_option).lower()
                    path_vdb            = path_folder_vdb + folder_name + '/'
                    if count_files(path_vdb) == 0:
                        documents           = []
                        directory           = path_folder_datasource + folder_name + '/'
                        for f in os.listdir(directory):
                            try:
                                print("::::::Loading File:::", directory + f)
                                docs        = getPdfDocuments(directory + f)
                                documents.extend(docs)
                            except:
                                print("Error while loading document:", f)
                        chunks              = getTextSplit(documents, 4096, 256)
                        vectorstore         = getChromsDB(chunks, gblVariables.embeddings, path_vdb)
                    #    gblVariables.retriever           = vectorstore.as_retriever()
                    else:
                        vectorstore         = getChromsDB(None, gblVariables.embeddings, path_vdb)
                    gblVariables.retriever  = vectorstore.as_retriever()

            gblVariables.prev_selected_option   = selected_option

    with right:
        with st.container(height=540, border=True, key="bottom"):
            contextualize_q_system_prompt=(
                "Given a chat history and the latest user question"
                "which will have reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
            
            print("::gblVariables.llm::", gblVariables.llm)
            print("::gblVariables.retriever::", gblVariables.retriever)
            history_aware_retriever=create_history_aware_retriever(gblVariables.llm, 
                                                                gblVariables.retriever,
                                                                contextualize_q_prompt)
            print("History Aware Finish")
            ## Answer question

            # Answer question
            system_prompt = (
                "You are in conversation with a human, with the primary goal of answering their questions. Use the uploaded "
                "documents as the main source of information, but supplement with your base knowledge if the documents don't "
                "cover the query, explicitly stating so. Respond courteously to greetings like 'Hello' or 'Good morning.' "
                "If you don't know an answer, acknowledge it. Keeping responses in bullet points and between "
                "300-400 words unless otherwise specified, provide answer to given question: "
                "\n\n"
                "{context}"
                )
            qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
            
            question_answer_chain   = create_stuff_documents_chain(gblVariables.llm,qa_prompt)
            rag_chain               = create_retrieval_chain(history_aware_retriever,question_answer_chain)

            st.markdown(
                """
                <style>
                    .st-emotion-cache-1c7y2kd {
                        flex-direction: row-reverse;
                        text-align: right;
                    }
                </style>
                """,
                    unsafe_allow_html=True,
                )
            def get_session_history(session_id:str)->BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id]  = ChatMessageHistory()
                return st.session_state.store[session_id]
            
            conversational_rag_chain= RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key  = "input",
                history_messages_key= "chat_history",
                output_messages_key = "answer"
            )

            session_history = get_session_history(session_id)

            for message in session_history.messages:
                if message.type == 'human':
                    with st.chat_message('user'):
                        st.markdown(message.content)
                else:
                    with st.chat_message('assistant'):
                        st.markdown(message.content)
            
            contr_prompt = right.container()
            with contr_prompt:
                prompt = st.chat_input("Please type your question.")
            contr_prompt.float("bottom:50;background-color: white;")

            if prompt:
                
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    response = conversational_rag_chain.invoke(
                        {"input": prompt},
                        config={
                            "configurable": {"session_id":session_id}
                        },
                    )
                    st.markdown(response['answer'])
                    
    return True