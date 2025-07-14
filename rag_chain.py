from retriever import DocumentRetriever
from generator import generate_answer_openai

def run_rag_pipeline(query, retriever, index, texts, top_k=3):
    query_vec = retriever.model.encode([query])
    D, I = index.search(query_vec, top_k)

    context_chunks = [texts[i] for i in I[0]]
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant answering based on the following document context.

Context:
{context}

Question: {query}
Answer:
    """.strip()

    return generate_answer_openai(prompt)
