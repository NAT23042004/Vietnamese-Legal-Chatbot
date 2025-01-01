import  os
import requests
import streamlit as st

CHATBOT_URL = os.getenv("CHATBOT_URL")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        Đây là giao diện chatbot với [Langchain] (https://python.langchain.com/docs/get_started/introduction)
        agent được thiết kế để trả lời các câu hỏi liên quan đến pháp luật Việt Nam trong lĩnh vực khoa học và
        công nghệ. 
        Agent này sử dụng [Hybrid RAG] (https://arxiv.org/abs/2408.04948) kết hợp giữa [Graph RAG] (https://microsoft.github.io/graphrag/) 
        và [Vector RAG] () để truy vấn đối với dữ liệu có cấu trúc lẫn không có câu trúc đã được tổng hợp từ [Neo4j] (https://neo4j.com/).
        """
    )
st.title("Vietnamese Legal Chatbot")
st.info("Nhiệm vụ của tôi là trả lời các câu hỏi về pháp luật Việt Nam trong lĩnh vực khoa học và công nghệ ")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])
        if "explanation" in message.keys():
            with st.status("Giải thích", state="Xong"):
                st.info(message["explanation"])


if prompt := st.chat_input("Bạn muốn biết về điều gì?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text" : prompt}

    with st.spinner("Đang tìm kiếm câu trả lời phù hợp..."):
        response = requests.post(CHATBOT_URL, json=data)

        if response.status_code == 200:
            output_text = response.json()["output"]
            explanation = response.json()["intermediate_steps"]

        else:
            output_text = """Trong quá trình xử lý câu hỏi của bạn đã xảy ra lỗi.
             Vui lòng hãy thử lại hoặc diễn tả lại câu hỏi của bạn theo một cách khác rõ ràng hơn."""
            explanation = output_text

        st.chat_message("assistant").markdown(output_text)
        st.status("Giải thích", state="Xong").info(explanation)

        st.session_state.messages.append(
            {
            "role" : "assistant",
            "output": output_text,
            "explanation": explanation,
            }
        )