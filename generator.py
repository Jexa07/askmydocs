from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("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"))

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
