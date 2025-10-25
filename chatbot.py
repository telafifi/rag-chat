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

# Initialize the HuggingFace sentence-transformer model for creating embeddings from document chunks.
# Embeddings are numerical vector representations of text that enable efficient similarity search and semantic comparisons in a vector database.
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the Anthropic Claude model for generating responses.
# Claude is a powerful language model that can understand and generate natural language text.
llm = ChatAnthropic(temperature=0.5, model='claude-3-haiku-20240307')

# Connect to the ChromaDB vector store.
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever.
# This is done to retrieve the relevant chunks based on the question asked.
# It is useful for retrieval and analysis.
num_results = 10
retriever = vector_store.as_retriever(
    search_kwargs={'k': num_results}
)

# Call this function for every message added to the chatbot.
# This is done to generate a response to the user's question.
# It is useful for retrieval and analysis.
def stream_response(message, history):
    # Retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)
    
    # Debug: print retrieved chunks
    for i, doc in enumerate(docs[:3]):  # Show first 3 chunks
        filename = doc.metadata.get('filename', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'

    # Add all the chunks to 'knowledge' with source information.
    # This is done to store the chunks in the knowledge base.
    # It is useful for retrieval and analysis.
    knowledge = ""
    sources = set()

    # This loop prepares and organizes the retrieved document chunks (the "knowledge base" for the answer).
    # It does so by collecting relevant chunks in the variable `knowledge`, and aggregating their source filenames in `sources`.
    # This enables the chatbot to not only provide relevant information in its final answer,
    # but also to cite or mention which documents the information comes from, improving traceability and trust in the RAG system.
    for doc in docs:
        # Add source information
        if hasattr(doc, 'metadata') and 'filename' in doc.metadata:
            filename = doc.metadata['filename']
            sources.add(filename)
            knowledge += f"[From: {filename}]\n{doc.page_content}\n\n"
        else:
            knowledge += doc.page_content+"\n\n"


    # This section is reserved for generating and returning the chatbot's final answer.
    # Specifically, it is where you generate the LLM-based response using all the gathered "knowledge" (context),
    # formatting or processing as needed before returning the output to the chat interface.
    # Any logic for answer construction, streaming, or post-processing should be placed here.
    if message is not None:

        partial_message = ""

        if knowledge.strip():
            sources_list = ", ".join(sources) if sources else "the documents"

            # This is the prompt that is sent to the LLM.
            # It is used to generate the response to the user's question.
            # It is useful for retrieval and analysis.
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

        # Stream the response to the chat interface.
        # This is done to generate a response to the user's question.
        for chunk in llm.stream(rag_prompt):
            if hasattr(chunk, 'content') and chunk.content:
                partial_message += chunk.content
                yield partial_message

# initiate the Gradio app for a chat interface.
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()