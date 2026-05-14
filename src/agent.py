from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.config import settings
from src.retriever import retrieve_context


SYSTEM_PROMPT = """
You are a grounded RAG assistant. Answer only using the retrieved context.
If the context is insufficient, say what is missing instead of guessing.
"""


def generate_answer(question: str):
    docs = retrieve_context(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.openai_api_key)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "sources": [doc.metadata for doc in docs],
    }
