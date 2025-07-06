from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import RetrievalQA
from colorama import Fore, Back, Style


# 1. Load the PDF
loader = PyPDFLoader("Raj Beladiya Resume.pdf")
pages = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
docs = splitter.split_documents(pages)

# store plain text for keyword search
all_text_chunks = [doc.page_content for doc in docs]

# 3. Create local embeddings
embeddings = OllamaEmbeddings(model="mistral")
db = Chroma.from_documents(docs, embeddings, persist_directory="./db")

# 4. Build retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# 5. Connect to local LLM
llm = OllamaLLM(model="mistral")


# 6. Build QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def hybrid_retriever(query, vectorstore, text_chunks, k=3):
    # vector retrieval
    vector_hits = vectorstore.similarity_search(query, k=k)

    # keyword retrieval (case-insensitive)
    keyword_hits = []
    query_words = query.lower().split()
    for chunk in text_chunks:
        if any(word in chunk.lower() for word in query_words):
            keyword_hits.append(chunk)

    # combine
    combined = set()
    for doc in vector_hits:
        combined.add(doc.page_content)
    for chunk in keyword_hits:
        combined.add(chunk)

    # return as a list of Document objects
    from langchain_core.documents import Document
    return [Document(page_content=text) for text in combined]


while True:
    query = input("Ask a question about the PDF (or 'exit'): ")
    if query.lower() == "exit":
        break

    # hybrid retrieval
    retrieved_chunks = hybrid_retriever(query, db, all_text_chunks, k=3)

    # assemble the text for the LLM
    context_text = "\n".join(chunk.page_content for chunk in retrieved_chunks)
    print()

    # prompt the LLM directly
    prompt = f"""Use the following document context to answer the question:

    {context_text}

    Question: {query}
    Answer:"""

    answer = llm.invoke(prompt)
    print("Answer:", answer)
