# Bring in dependencies
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import Gemini  # Replace OpenAI with Gemini
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

# Set up the API key for Gemini
os.environ['GEMINI_API_KEY'] = apikey

# App framework
st.title('YouTube Script Generator')
prompt = st.text_input('Plug in your prompt here') 

# Add a slider for the user to select the video length
video_length = st.slider('Select the video length (minutes)', min_value=1, max_value=30, value=10)

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research', 'video_length'], 
    template=('write me a youtube video script based on this title TITLE: {title} '
              'while leveraging this wikipedia research: {wikipedia_research}. '
              'The script should be suitable for a video that is {video_length} minutes long.')
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Llms
llm = Gemini(temperature=0.9)  # Use Gemini instead of OpenAI
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show content on the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research, video_length=video_length)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
