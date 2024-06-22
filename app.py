import streamlit as st
from llama_cpp import Llama

# Initialize the Llama model
llm = Llama(
    model_path="./Phi-3-mini-4k-instruct-q4.gguf",  # path to GGUF file
    n_ctx=4096,  # The max sequence length to use
    n_threads=8, # The number of CPU threads to use
    n_gpu_layers=35, # The number of layers to offload to GPU
)

def generate_response(prompt):
    output = llm(
        f"\n{prompt}\n",
        max_tokens=2000,  # Generate up to 256 tokens
        stop=[""], 
        echo=True,  # Whether to echo the prompt
    )
    return output['choices'][0]['text']

# Streamlit app layout
st.title("Travel assistant")

st.write("Ask me anything!")

# User input
user_input = st.text_input("You:", "")

# Generate response when user submits input
if user_input:
    with st.spinner("Thinking..."):
        response = generate_response(user_input)
    st.text_area("Llama:", response, height=200)

# Run the app with streamlit
#if __name__ == '__main__':
    #st.run()
