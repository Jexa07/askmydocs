from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("YOUR API KEY"))

def generate_answer_openai(prompt, model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error] {e}"
