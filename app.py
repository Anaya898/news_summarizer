import openai

# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = 'api-key'
def generate_response(prompt):
    response = openai.Completion.create(
        model="gpt-3.5-turbo",  # Use the latest model name
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    print("AI Chatbot is ready. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = generate_response(user_input)
        print(f"Bot: {response}")
