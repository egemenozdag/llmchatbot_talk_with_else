from src.chat_engine import process_message

def test_process_message():
    user_input = "What is the weather like?"
    processed = process_message(user_input)
    assert processed == user_input.lower().strip()
