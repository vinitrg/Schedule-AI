import streamlit as st
import requests
import json
import time
from typing import Dict, Any, Iterator

class AzureOpenAIChat:
    def __init__(self):
        self.API_ENDPOINT = st.secrets.get("AZURE_OPENAI_API_ENDPOINT", "")
        self.API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY", "")  

        if not self.API_KEY:
            raise ValueError("Azure OpenAI API Key is missing.")

    def generate_response_stream(
        self,
        query: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> Iterator[str]:
        """Generate streaming response from Azure OpenAI"""
        headers = {
            "Content-Type": "application/json",
            "api-key": self.API_KEY,
        }

        data = {
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": True  # Enable streaming
        }

        try:
            response = requests.post(
                self.API_ENDPOINT,
                headers=headers,
                json=data,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line.strip() == b'data: [DONE]':
                    break
                if line.startswith(b'data: '):
                    json_str = line[6:].decode('utf-8')
                    try:
                        json_data = json.loads(json_str)
                        if 'choices' in json_data and len(json_data['choices']) > 0:
                            delta = json_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except requests.exceptions.RequestException as req_err:
            raise RuntimeError(f"Request error: {req_err}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {str(e)}")


def main():
    st.set_page_config(page_title="OpenAI Playground", page_icon="💬")
    st.title("OpenAI Playground")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Enter your message"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display streaming response
        chat_client = AzureOpenAIChat()

        # Create a placeholder for the streaming response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                for text_chunk in chat_client.generate_response_stream(
                    prompt,
                    max_tokens=1000,
                ):
                    full_response += text_chunk
                    # Update response in real-time
                    response_placeholder.markdown(full_response + "▌")

                # Final update without cursor
                response_placeholder.markdown(full_response)

                # Add assistant's message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                })

            except RuntimeError as err:
                st.error(f"Error generating response: {err}")
                response_placeholder.markdown("Sorry, I couldn't generate a response.")


if __name__ == "__main__":
    main()
     # Footer with text and link
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 10px;
            right: 10px;
            font-size: 14px;
            color: #666;
            text-align: right;
            z-index: 1000;
        }
        .footer a {
            color: #007BFF;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        </style>
        <div class="footer">
            By: <a href="https://www.linkedin.com/in/vinit-gujarathi/" target="_blank">Vinit Gujarathi</a>
        </div>
        """,
        unsafe_allow_html=True,
    )