import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL issues for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot response function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Save chat history
def save_chat(user_input, chatbot_response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([user_input, chatbot_response, timestamp])

# Display chat messages in a styled format
def display_chat(user_msg, bot_msg):
    with st.container():
        st.markdown(f'<div style="text-align: right; background-color: #d1e7dd; padding: 10px; border-radius: 10px; margin-bottom: 5px;">{user_msg}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: left; background-color: #f8d7da; padding: 10px; border-radius: 10px;">{bot_msg}</div>', unsafe_allow_html=True)

# Typing animation effect for chatbot
def chatbot_typing(response):
    display_text = ""
    message_container = st.empty()
    for char in response:
        display_text += char
        message_container.text(display_text)
        time.sleep(0.05)

# Main function
def main():
    st.title("🤖 Chatbot using NLP")

    # Sidebar menu
    menu = ["Chat", "Conversation History", "About"]
    choice = st.sidebar.radio("Menu", menu)

    # Chat Interface
    if choice == "Chat":
        st.write("💬 Type your message below and press Enter to chat.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous messages
        for msg in st.session_state.messages:
            with st.chat_message("user" if msg["is_user"] else "assistant"):
                st.markdown(msg["text"])

        # User input box
        user_input = st.text_input("You:")

        if user_input:
            chatbot_response = chatbot(user_input)

            # Save messages in session state
            st.session_state.messages.append({"text": user_input, "is_user": True})
            st.session_state.messages.append({"text": chatbot_response, "is_user": False})

            # Display chatbot response with animation
            with st.chat_message("assistant"):
                chatbot_typing(chatbot_response)

            # Save to conversation history
            save_chat(user_input, chatbot_response)

    # Conversation History
    elif choice == "Conversation History":
        st.header("📜 Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.markdown(f"**User:** {row[0]}  \n**Chatbot:** {row[1]}  \n*{row[2]}*")
                    st.markdown("---")
        else:
            st.write("No conversation history available.")

    # About Page
    elif choice == "About":
        st.subheader("📖 About This Chatbot")
        st.write("Welcome to our **AI-powered chatbot**! This chatbot is designed to **understand and respond** to user queries using **Natural Language Processing (NLP)**. It is trained with **machine learning models** to provide relevant and accurate responses.")

        st.subheader("🔹 Key Features:")
        st.markdown("""
        ✅ **Built with NLP:** Understands human language and responds intelligently.  
        ✅ **Machine Learning-based:** Uses **Logistic Regression** for intent recognition.  
        ✅ **User-friendly Interface:** Powered by **Streamlit** for an interactive experience.  
        ✅ **Chat History Feature:** Keeps a record of past conversations.  
        """)

        st.subheader("🛠️ Technologies Used:")
        st.markdown("""
        - **Python** for backend development  
        - **NLTK** for text processing  
        - **TF-IDF Vectorization** for feature extraction  
        - **Logistic Regression** for intent classification  
        - **Streamlit** for the chatbot interface  
        """)

        st.write("This chatbot can be extended by integrating **deep learning models**, **API support**, or adding **more intents for enhanced responses**. 🚀")


if __name__ == '__main__':
    main()
