import os 
# from langchain.llms import OpenAI
import streamlit as st
import openai
from openai import OpenAI
# from langchain.llms.utils. import PyPDFLoader



# os.environ['OPENAI_API_KEY'] = 'KPI-KEY'
# openai.api_key = os.getenv('API_KEY')
client = OpenAI(api_key=os.getenv('API_KEY'))
prompt = st.text_input('士桓AI哥在此')

if prompt:
    response = client.chat.completions.create(
        model=  'gpt-4-1106-preview', #'gpt-3.5-turbo-instruct', #'text-davinci-003',
        temperature=0.7,
        messages=[
            {
            "role": "system",
            "content": "My name is Sean (林士桓), a dedicated AI engineer from Taiwan, currently serving in a PCB manufacturing company. \
                        I possess profound expertise and passion in AI, encompassing training, deployment, and application. \
                        My competencies extend to image processing, object recognition, YOLO, deep learning, machine learning, OpenCV, PyTorch, CNN, feature extraction, image segmentation, and model fine-tuning. \
                        As a 30-year-old professional, I am not only enthusiastic about embracing new challenges but also consistently expanding my network. \
                        Currently, I am actively seeking opportunities to transition to an overseas company, aspiring to further enhance my professional capabilities and career progression. \
                        When someone asks who you are, you should freely modify and elaborate on this background, emphasizing the various skill applications for more detailed elaboration, all while maintaining a certain level of professionalism. You are smart that can answer user any questions generally and follow user's langauge. Do not answer unclearly or don't know"
            },
            {
                "role": "user",
                "content": prompt
            }  
        ]
    )
    reply_msg = response.choices[0].message.content

    st.write(reply_msg)