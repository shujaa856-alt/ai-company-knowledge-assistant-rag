def build_context(retrieved_chunks):
    context_parts = []

    for index, chunk in enumerate(retrieved_chunks, start=1):

        context = f"""
Context {index}

{chunk["document"]}
"""

        context_parts.append(context.strip())

    return "\n\n".join(context_parts)