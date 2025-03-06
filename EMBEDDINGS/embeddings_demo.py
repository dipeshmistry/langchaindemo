

import os
from mistralai import Mistral

api_key = os.getenv("MISTRAL_API_KEY")
print(api_key)
model = "mistral-large-latest"

client = Mistral(api_key=api_key)
question = input("Type Your Question :   ")
chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": question,
        },
    ]
)
print(chat_response.choices[0].message.content)