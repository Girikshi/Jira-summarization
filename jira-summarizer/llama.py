import gradio as gr
import requests
import os
import re

model_access_key = "m8rwie5pevcltiebfn9x74gyv2ajv649"

cml_model_service = "https://modelservice.ml-e95c7ec4-6c3.eng-hack.vnu8-sqze.cloudera.site/model"

model_request_json_template = '{"accessKey":"%s","request":{"prompt":"%s"}}'

# Adapted from https://colab.research.google.com/drive/1SSv6lzX3Byu50PooYogmiwHqf5PQN68E?usp=sharing#scrollTo=PeEh17FDLzEe


SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are a helpful CML Hackathon training bot. The training takes place on Sept 8th and is presented by ROBO-ALEX. Your answers are clear and concise. Dont use emojis.
<</SYS>>
"""


# Formatting function for message and history
def format_message(message: str, history: list, memory_limit: int = 3) -> str:
    """
    Formats the message and history for the Llama model.
    Parameters:
        message (str): Current message to send.
        history (list): Past conversation history.
        memory_limit (int): Limit on how many past interactions to consider.
    Returns:
        str: Formatted message string
    """
    # always keep len(history) <= memory_limit
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        return SYSTEM_PROMPT + f"{message} [/INST]"

    formatted_message = SYSTEM_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"

    # Handle conversation history
    for user_msg, model_answer in history[1:]:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"

    # Handle the current message
    formatted_message += f"<s>[INST] {message} [/INST]"

    return formatted_message


def llm_response(message, chat_history):
    """
    Calls out to CML Model for LLM responses
    """

    prompted_msg = format_message(message, chat_history)
    json_esc_prompted_msg = prompted_msg.replace("\n", "\\n")

    data = model_request_json_template % (model_access_key, json_esc_prompted_msg)
    r = requests.post(cml_model_service,
                      data=data,
                      headers={'Content-Type': 'application/json'})

    prediction = r.json()["response"]
    print('--- Prompt sent to model')
    print(prompted_msg)
    print('--- Prediction returned from model')
    print(prediction)
    # bot_message is everything after the final [/INST] in the response
    start_of_bot = prediction.rfind('[/INST]')
    bot_message = prediction[start_of_bot + 7:]

    return bot_message


demo = gr.ChatInterface(llm_response)
demo.launch(server_port=int("5000"),
            enable_queue=True)

# Run the following in the session workbench to kill the gradio interface. Hit play button to rerun the script

# demo.close()