from api_handler import call_llm_api
from unittest.mock import Mock

def test_call_llm_api():
    mock_response = {"choices": [{"text": "Hello!"}]}
    mock_api = Mock(return_value=mock_response)
    response = call_llm_api("Hi", api_client=mock_api)
    assert response == "Hello!"
