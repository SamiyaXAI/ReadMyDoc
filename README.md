# ReadMyDoc
ReadMyDoc is a simple Document Question-Answering app built with Streamlit, LangChain, and
OpenAI GPT.
You can upload a document and ask questions about its content -- the app will give you smart
answers based on the document.
Features
- Upload Documents (.pdf or .txt)
- Ask Questions about your uploaded document
- Secure OpenAI API Key input (not stored)
- Powered by RAG (Retrieval-Augmented Generation)
- Streamlit UI for easy use
Installation
1. Clone this repository
2. Install dependencies
 pip install -r requirements.txt
3. Run the app
 streamlit run app.py
OpenAI API Key
You'll need an OpenAI API key to use the app.
Enter it directly in the app -- your key is never stored or shared.
How to Use
1. Start the app using the command above
2. Upload a document (.pdf or .txt)
3. Enter your question in the text box
4. Provide your OpenAI API key
5. Click Submit to get your answer!
Security Note
Your API key is entered via a text input field and used only during the session.
The app does not log, store, or transmit your API key anywhere.
