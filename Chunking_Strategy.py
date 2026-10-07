
import os
import numpy as np

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# --------------------------------------------------
# 1. Input document (No PDF required)
# --------------------------------------------------

text = """
Document: Azure Virtual Network Guide

Azure VNet allows Azure resources to communicate securely.

Network Security Groups (NSGs) control inbound and outbound traffic.

Azure Private Endpoints provide private connectivity to Azure services.
"""

# Create a LangChain Document directly from text
document = Document(
    page_content=text,
    metadata={"source": "Azure Virtual Network Guide"}
)


# --------------------------------------------------
# 2. Load embedding model (shared by all strategies)
# --------------------------------------------------

print("Loading HuggingFace Embeddings model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 3. Strategy 1: Fixed-size chunking
# --------------------------------------------------

def fixed_size_chunking(doc):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        separators=[" ", ""]
    )

    return splitter.split_documents([doc])


# --------------------------------------------------
# 4. Strategy 2: Overlapping chunking
# --------------------------------------------------

def overlapping_chunking(doc):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=30,
        separators=[" ", ""]
    )

    return splitter.split_documents([doc])


# --------------------------------------------------
# 5. Strategy 3: Recursive chunking
# --------------------------------------------------

def recursive_chunking(doc):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_documents([doc])


# --------------------------------------------------
# 6. Strategy 4: Semantic chunking
# --------------------------------------------------

def semantic_chunking(doc, similarity_threshold=0.5):
    # Split the document into sentences
    sentences = [
        s.strip()
        for s in doc.page_content.replace("\n", " ").split(".")
        if s.strip()
    ]

    # Generate an embedding for each sentence
    vectors = embeddings.embed_documents(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        v1 = np.array(vectors[i - 1])
        v2 = np.array(vectors[i])

        # Calculate cosine similarity
        similarity = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )

        print(
            f"Similarity between sentences {i} and {i+1}: "
            f"{similarity:.4f}"
        )

        # Low similarity indicates a possible topic boundary
        if similarity < similarity_threshold:
            chunks.append(
                Document(
                    page_content=". ".join(current_chunk) + ".",
                    metadata=doc.metadata.copy()
                )
            )

            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    # Add the last chunk
    if current_chunk:
        chunks.append(
            Document(
                page_content=". ".join(current_chunk) + ".",
                metadata=doc.metadata.copy()
            )
        )

    return chunks


# --------------------------------------------------
# 7. Run all strategies and store in ChromaDB
# --------------------------------------------------

def run_pipeline():

    strategies = {
        "fixed_size": fixed_size_chunking,
        "overlapping": overlapping_chunking,
        "recursive": recursive_chunking,
        "semantic": semantic_chunking
    }

    query = "How do NSGs control traffic?"

    for name, strategy in strategies.items():

        print("\n" + "=" * 60)
        print(f"CHUNKING STRATEGY: {name.upper()}")
        print("=" * 60)

        # Create chunks
        chunks = strategy(document)

        print(f"\nTotal chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i + 1}:")
            print(chunk.page_content)

        # Store each strategy in a separate Chroma collection
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"azure_vnet_{name}",
            persist_directory=f"./chroma_db/{name}"
        )

        # Similarity search
        results = vector_db.similarity_search(
            query,
            k=1
        )

        print("\n--- Retrieval Result ---")
        print("Query:", query)

        if results:
            print("Retrieved chunk:")
            print(results[0].page_content)
            print("Metadata:", results[0].metadata)


if __name__ == "__main__":
    run_pipeline()