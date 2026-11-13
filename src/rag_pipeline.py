from pathlib import Path
import re

import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline


DATA_FILE = Path("data/azure_aws_cloud_rag_training_20_pages.txt")
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "azure_aws_cloud"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def clean_text(text):
    text = text.replace("\t", " ")
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_vector_database():
    loader = TextLoader(str(DATA_FILE), encoding="utf-8")
    documents = loader.load()

    for document in documents:
        document.page_content = clean_text(document.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)
    chunk_texts = [chunk.page_content for chunk in chunks]

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedding_model.encode(chunk_texts)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [str(i) for i in range(len(chunks))]
    metadatas = [
        {
            "source": chunk.metadata.get("source", str(DATA_FILE)),
            "chunk_number": i,
        }
        for i, chunk in enumerate(chunks)
    ]

    collection.upsert(
        ids=ids,
        documents=chunk_texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Vectors stored: {collection.count()}")

    return collection, embedding_model


def retrieve(collection, embedding_model, question, top_k=3):
    question_embedding = embedding_model.encode([question])

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=top_k,
    )

    return results["documents"][0]


def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using only the context below.

Give one clear and concise answer.
Do not repeat the answer.
If the answer is not in the context, say:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}
"""

    llm = pipeline(
        "text-generation",
        model=GENERATION_MODEL,
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = llm(
        messages,
        max_new_tokens=180,
        do_sample=False,
    )

    generated = response[0]["generated_text"]

    if isinstance(generated, list):
        return generated[-1]["content"]

    return generated


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing training document: {DATA_FILE}\n"
            "Place the Azure/AWS training TXT file in the data folder."
        )

    collection, embedding_model = build_vector_database()

    question = "How does Azure provide private connectivity to storage?"

    retrieved_chunks = retrieve(
        collection,
        embedding_model,
        question,
        top_k=3,
    )

    answer = generate_answer(
        question,
        retrieved_chunks,
    )

    print("\nQUESTION:")
    print(question)

    print("\nRETRIEVED CHUNKS:")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)

    print("\nANSWER:")
    print(answer)


if __name__ == "__main__":
    main()
