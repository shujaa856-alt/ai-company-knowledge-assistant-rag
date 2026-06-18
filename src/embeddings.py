from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence = "Employees get 20 vacation days every year."

embedding = model.encode(sentence)

print("Embedding Length:", len(embedding))