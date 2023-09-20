from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from output_parsers import jira_parser
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def predictor(jira_id):
    pd.options.display.max_colwidth = 10000
    openai_api_key='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'

    df = pd.read_json('sample_comments/data.json', encoding="utf8")

    selected_rows = df[df['Issue key'] == jira_id]
    selected_row_id = selected_rows['Issue key'].to_string(index=False)
    selected_row_assignee = selected_rows['Assignee name'].to_string(index=False)
    selected_row_status = selected_rows['Status'].to_string(index=False)
    selected_row_description = selected_rows['Description'].to_string(index=False)
    selected_row_comments = selected_rows['Comments'].to_string(index=False)

    print(selected_row_description)
    print(selected_row_comments)

    text = "Jira Id: " + selected_row_id + "\nAssignee Name: " + selected_row_assignee + "\nStatus: " + selected_row_status + \
        "\nDescription of the Jira: " + selected_row_description + "\nJira Comments: " + selected_row_comments
    summaries = text
    print(text)
    # Convert it back to a document
    summaries = Document(page_content=summaries)

    llm4 = ChatOpenAI(temperature=0,
                      openai_api_key=openai_api_key,
                      max_tokens=3000,
                      model='gpt-4-32k-0613',
                      request_timeout=120
                      )
    
    combine_prompt = """
    You will be given a series of comments of a jira.
    Your goal is to give detailed summary of all the comments in the jira.
    The reader should be able to grasp what happened in the jira.
    

    ```{text}```
    jira id:
    title:
    assignee:
    status:
    summary:
    \n{format_instructions}
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt,
                                             input_variables=["text"],
                                             partial_variables={
                                                 "format_instructions": jira_parser.get_format_instructions()
                                             },
                                             )

    reduce_chain = load_summarize_chain(llm=llm4,
                                        chain_type="stuff",
                                        prompt=combine_prompt_template,
                                        )

    output = reduce_chain.run([summaries])
    print(output)

    return jira_parser.parse(output)