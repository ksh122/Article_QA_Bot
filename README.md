# Article_QA_Bot
Article_QA_Bot is a user friendly designed for retrieval of data from a bunch of news article links provided to it.

![image](https://github.com/user-attachments/assets/a1deb706-44bf-444f-83fb-8817e1c7f127)

## Features

* Load URLs or upload text files containing URLs to fetch article content.
* Process article content through LangChain's UnstructuredURL Loader
* Construct an embedding vector using OpenAI's embeddings and leverage FAISS, a powerful similarity search library, to enable swift and effective retrieval of 
  relevant information
* Interact with the LLM's (Chatgpt) by inputting queries and receiving answers along with source URLs.

## Installation
1. Clone this repository to your local machine using:
   git clone https://github.com/ksh122/Article_QA_Bot 

2. Install the required dependencies using pip:
   pip install -r requirements.txt

3. Set up your API key by creating a .env file in the project root and adding your API
   GROQ_API_KEY = your_api_key_here

4. Run the Streamlit app by executing:
   streamlit run main.py
