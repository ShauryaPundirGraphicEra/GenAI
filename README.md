# Project 1

# QnA Chatbot with Streamlit, Hugging Face, and LangChain

A web-based application that allows users to ask questions or summarize text using Hugging Face's language models integrated with LangChain and presented through a Streamlit interface.

## Project Overview
The QnA Chatbot leverages **Streamlit** for an interactive UI, **Hugging Face's Inference API** for powerful language models, and **LangChain** for prompt engineering. Users can input questions or text, select tasks (QnA or summarization), and adjust model parameters like temperature and max tokens. This project demonstrates AI integration with a web framework, making it ideal for learning or showcasing in a portfolio.

## Features
- **Interactive UI**: Built with Streamlit, featuring a text input area, sidebar for model selection, and parameter tuning (temperature: 0.1–1.0, max tokens: 50–500).
- **Task Flexibility**: Supports question-answering (e.g., "What is the theory of relativity?") and text summarization (e.g., "Summarize this article about AI").
- **Hugging Face Integration**: Uses models like `mistralai/Mixtral-8x7B-Instruct-v0.1` via the Inference API, configured with a secure API token.
- **Prompt Engineering**: Employs LangChain's `PromptTemplate` for structured, context-appropriate responses.
- **Error Handling**: Manages API errors, invalid inputs, and model issues with user-friendly feedback.
- **Extensibility**: Ready for enhancements like file uploads or multi-turn conversations.

## Tech Stack
- **Streamlit**: Web interface and deployment.
- **LangChain**: Prompt management and model interaction.
- **Hugging Face**: Language models via Inference API.
- **Python Libraries**: `streamlit`, `langchain-huggingface`, `python-dotenv`.
- **Python Version**: 3.10 or higher.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/qna-chatbot.git
   cd qna-chatbot

2. **Install Dependencies:**:
   ```bash
   pip install streamlit langchain-huggingface python-dotenv

3. **Set Up Environment:**:
- Create a .env file in the project root.
- Add your Hugging Face API token

   ```bash
   HUGGINGFACEHUB_API_TOKEN=your_token_here

 - Get a token from Hugging Face.

4. **Run the App Locally:**:
   ````bash
    streamlit run app.py
## Demo

![image](https://github.com/user-attachments/assets/67fbfb26-b85f-465e-b0f2-6b399c32b2d0)
