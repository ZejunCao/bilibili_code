import streamlit as st
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, Qwen2ForCausalLM
from threading import Thread
device = "cuda"  # the device to load the model onto


st.title("大模型流式输出测试")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个人工智能助手"}]

    model_path = 'D:\learning\python\pretrain_checkpoint\Qwen2.5-1.5B-Instruct'
    st.session_state.model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_path)

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 
if prompt := st.chat_input("请输入问题"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    text = st.session_state.tokenizer.apply_chat_template(st.session_state.messages, tokenize=False, add_generation_prompt=True)
    model_inputs = st.session_state.tokenizer(text, return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(st.session_state.tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=1024)
    # 在单独的线程中调用.generate()
    thread = Thread(target=st.session_state.model.generate, kwargs=generation_kwargs)
    thread.start()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            message_placeholder.markdown(generated_text)
    st.session_state.messages.append({"role": "assistant", "content": generated_text})

if st.button("清空"):
    st.session_state.messages = [{"role": "system", "content": "你是一个人工智能助手"}]
    st.rerun()