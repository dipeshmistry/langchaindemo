import os
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"

client = Mistral(api_key=api_key)

input = input("Enter a text.")

embeddings_batch_response = client.embeddings.create(
    model=model,
    inputs=[input],
)
print(embeddings_batch_response.data[0].embedding)