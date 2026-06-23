import time

from retriever import retrieve
from context_builder import build_context
from llm import generate_response
from chatbot import PROMPT_TEMPLATE

def evaluate(question):
    """
    Evaluate the complete RAG pipeline.

    Args:
        question (str): 
            User's question.

    Returns:
        dict: 
            Evaluation data including timings, retrieved chunks, and generated answer.
    """     

    # --------------------------------
    # Measure retrieval time   
    # --------------------------------
    retrieval_start = time.perf_counter()

    retrieved_chunks = retrieve(question)

    retrieval_time = time.perf_counter() - retrieval_start


    # --------------------------------
    # Build Context
    # --------------------------------
    context = build_context(retrieved_chunks)


    # --------------------------------
    # Build Prompt
    # --------------------------------
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )


    # --------------------------------
    # Measure LLM Time
    # --------------------------------
    llm_start = time.perf_counter()

    answer = generate_response(prompt)

    llm_time = time.perf_counter() - llm_start


    # --------------------------------
    # Total Time
    # --------------------------------
    total_time = retrieval_time + llm_time


    # --------------------------------
    # Print Evaluation Report
    # --------------------------------
    print("=" * 80)
    print("RAG EVALUATION REPORT")
    print("=" * 80)

    print(f"\nQuestion: {question}")

    print("\n" + "-" * 80)
    print("Retrieved Chunks")
    print("-" * 80)

    print(f"Total Chunks Retrieved: {len(retrieved_chunks)}")

    for index, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\nChunk {index}")
        print(f"ID: {chunk['id']}")
        print(f"Distance: {chunk['distance']:.4f}")

    print("\n" + "-" * 80)
    print("Performance")
    print("-" * 80)

    print(f"\nRetrieval Time : {retrieval_time:.4f} sec")
    print(f"LLM Time         : {llm_time:.4f} sec")
    print(f"Total Time       : {total_time:.4f} sec")

    print("\n" + "-" * 80)
    print("Generated Answer")
    print("-" * 80)

    print(answer)

    print("\n" + "=" * 80)

    # --------------------------------
    # Return Structured Results
    # --------------------------------
    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_time": retrieval_time,
        "llm_time": llm_time,
        "total_time": total_time,
        "answer": answer,
    }
if __name__ == "__main__":

    while True:

        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        evaluate(question)