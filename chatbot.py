from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# initiate the model
llm = ChatAnthropic(temperature=0.5, model='claude-3-haiku-20240307')

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = 10
retriever = vector_store.as_retriever(
    search_kwargs={'k': num_results}
)

# call this function for every message added to the chatbot
def stream_response(message, history):
    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)
    
    # Debug: print retrieved chunks
    print(f"Retrieved {len(docs)} chunks for query: {message}")
    for i, doc in enumerate(docs[:3]):  # Show first 3 chunks
        filename = doc.metadata.get('filename', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
        print(f"Chunk {i+1} (from {filename}): {doc.page_content[:100]}...")

    # add all the chunks to 'knowledge' with source information
    knowledge = ""
    sources = set()

    for doc in docs:
        # Add source information
        if hasattr(doc, 'metadata') and 'filename' in doc.metadata:
            filename = doc.metadata['filename']
            sources.add(filename)
            knowledge += f"[From: {filename}]\n{doc.page_content}\n\n"
        else:
            knowledge += doc.page_content+"\n\n"


    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        if knowledge.strip():
            sources_list = ", ".join(sources) if sources else "the documents"
            rag_prompt = f"""
            You are a helpful assistant that answers questions about the provided documents. 
            Use the information from the documents to answer the user's question naturally and conversationally.
            When relevant, you can mention which document(s) the information comes from.
            
            Question: {message}
            
            Conversation history: {history}
            
            Document content from {sources_list}: {knowledge}
            
            Please provide a helpful answer based on the document content above.
            """
        else:
            rag_prompt = f"""
            You are a helpful assistant. The user asked: {message}
            
            Unfortunately, I don't have access to the specific document content needed to answer this question. 
            Please try asking about topics that might be covered in the document, such as:
            - The paper's title and authors
            - The main contributions
            - Technical details about the methodology
            - Experimental results
            - Abstract or introduction content
            
            Conversation history: {history}
            """

        #print(rag_prompt)

        # stream the response to the Gradio App
        for chunk in llm.stream(rag_prompt):
            if hasattr(chunk, 'content') and chunk.content:
                partial_message += chunk.content
                yield partial_message

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()