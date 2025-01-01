import streamlit as st
import random
import time

import torch
from openai import OpenAI

st.title("大模型流式输出测试")

def get_response(message):
    openai_api_key = "EMPTY"
    openai_api_base = "http://0.0.0.0:5001/v1"  # 换成自己的ip+端口

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    response = client.chat.completions.create(
        model="qwen2.5",
        messages=message,
        stream=True,
    )

    for chunk in response:
        if chunk.choices[0].delta.content is None:
            yield ""
        else:
            yield chunk.choices[0].delta.content

if "messages" not in st.session_state:
	st.session_state.messages = [{"role": "system", "content": "你是一个人工智能助手"}]

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 
if prompt := st.chat_input("请输入问题"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        generated_text = ""
        for new_text in get_response(st.session_state.messages):
            generated_text += new_text
            message_placeholder.markdown(generated_text)
    st.session_state.messages.append({"role": "assistant", "content": generated_text})

if st.button("清空"):
    st.session_state.messages = [{"role": "system", "content": "你是一个人工智能助手"}]
    st.rerun()