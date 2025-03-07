import os
from mistralai import Mistral
import numpy as np

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"

client = Mistral(api_key=api_key)

input1 = input("Enter first text.")
input2 = input("Enter second text.")

embeddings_batch_response1 = client.embeddings.create(
    model=model,
    inputs=[input1],
)

embeddings_batch_response2 = client.embeddings.create(
    model=model,
    inputs=[input2],
)

similarity_score = np.dot(embeddings_batch_response1.data[0].embedding,embeddings_batch_response2.data[0].embedding)
print(similarity_score)
# print(embeddings_batch_response.data[0].embedding)