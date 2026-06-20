import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


def generate_response(prompt):

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text

if __name__ == "__main__":
    prompt = "Explain what Retrieval-Augmented Generation is in two sentences."

    response = generate_response(prompt)

    print(response)