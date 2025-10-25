from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# initiate the embeddings model
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# loading the PDF documents
loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()

print(f"Loaded {len(raw_documents)} documents from {DATA_PATH}")

# splitting the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
chunks = text_splitter.split_documents(raw_documents)

print(f"Created {len(chunks)} chunks from {len(raw_documents)} documents")

# Add metadata to chunks for better organization
for i, chunk in enumerate(chunks):
    # Extract filename from source
    if hasattr(chunk, 'metadata') and 'source' in chunk.metadata:
        filename = chunk.metadata['source'].split('/')[-1]
        chunk.metadata['filename'] = filename
        chunk.metadata['chunk_id'] = i

# creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]

# adding chunks to vector store
vector_store.add_documents(documents=chunks, ids=uuids)

print(f"Successfully added {len(chunks)} chunks to vector database")