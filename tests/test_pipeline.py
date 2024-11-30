from pipeline import chat_with_user

def test_chat_with_user():
    user_input = "Tell me a joke."
    response = chat_with_user(user_input)
    assert "joke" in response.lower()

def test_empty_input():
    user_input = ""
    response = chat_with_user(user_input)
    assert response == "I didn't understand your message. Please try again."
