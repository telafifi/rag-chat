from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Initialize the HuggingFace sentence-transformer model for creating embeddings from document chunks.
# Embeddings are numerical vector representations of text that enable efficient similarity search and semantic comparisons in a vector database.
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# The vector store is responsible for storing, indexing, and retrieving vector embeddings of document chunks.
# It enables fast similarity search by mapping text chunks (from PDFs) into vector representations (embeddings),
# and then persisting them in a local database (ChromaDB) for efficient retrieval during question answering.
# This makes it possible to quickly find relevant pieces of your documents based on semantic similarity to user queries.
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Loading the PDF documents
loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()

print(f"Loaded {len(raw_documents)} documents from {DATA_PATH}")

# Splitting the documents into chunks.
# This is done to create smaller, more manageable pieces of text that can be processed by the LLM.
# Chunking helps in reducing the complexity of the document and makes it easier for the LLM to process.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# Creating the chunks.
# This is done to create smaller, more manageable pieces of text that can be processed by the LLM.
# Chunking helps in reducing the complexity of the document and makes it easier for the LLM to process.
chunks = text_splitter.split_documents(raw_documents)

print(f"Created {len(chunks)} chunks from {len(raw_documents)} documents")

# Adding metadata to chunks for better organization.
# This is done to add additional information to the chunks that can be used for retrieval and analysis.
# You can expand metadata to include additional attributes for more powerful search and filtering.
# For example:
# - The filename (to identify which PDF the chunk is from)
# - The chunk ID (useful for traceability and debugging)
# - The original page numbers related to each chunk (for source attribution)
# - The document title or author, if extractable (for multi-document context)
# - Timestamps or processing dates (to know when the chunk was created)
# - Custom tags or categories (to enable topic-based queries)
# - Word/character counts (to support advanced filtering or analytics)
# This richer metadata can make your retrieval system more interpretable and customizable.
for i, chunk in enumerate(chunks):
    # Extract filename from source
    if hasattr(chunk, 'metadata') and 'source' in chunk.metadata:
        filename = chunk.metadata['source'].split('/')[-1]
        chunk.metadata['filename'] = filename
        chunk.metadata['chunk_id'] = i

# Creating unique ID's for each chunk.
# This is done to uniquely identify each chunk in the vector database.
# It is useful for retrieval and analysis.  
uuids = [str(uuid4()) for _ in range(len(chunks))]

# Adding chunks to vector store.
# This is done to store the chunks in the vector database.
# It is useful for retrieval and analysis.
vector_store.add_documents(documents=chunks, ids=uuids)

print(f"Successfully added {len(chunks)} chunks to vector database")