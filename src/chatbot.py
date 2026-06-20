from retriever import retrieve
from context_builder import build_context
from llm import generate_response


PROMPT_TEMPLATE = """
You are an AI Company Knowledge Assistant.

Answer the user's question ONLY using the provided context.

If the answer cannot be found in the provided context,
reply exactly:

"I couldn't find that information in the company documents."

Context:

{context}

Question:

{question}

Answer:
"""


def chat(question):
    retrieved_chunks = retrieve(question)

    print("\n" + "=" * 80)
    print("RETRIEVED CHUNKS")
    print("=" * 80)

    for index, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\nChunk {index}")
        print(f"ID: {chunk['id']}")
        print(f"Distance: {chunk['distance']:.4f}")
        print("-" * 80)
        print(chunk["document"])
        print("-" * 80)

    context = build_context(retrieved_chunks)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    print("\n" + "=" * 80)
    print("PROMPT SENT TO GEMINI")
    print("=" * 80)
    print(prompt)

    answer = generate_response(prompt)

    return answer


if __name__ == "__main__":

    while True:

        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        answer = chat(question)

        print("\nAnswer:\n")
        print(answer)