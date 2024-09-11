from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_groq import ChatGroq

def query_agent(data, query):

    df = pd.read_csv(data)
    llm = ChatGroq()

    agent = create_pandas_dataframe_agent(llm, df, verbose=True,allow_dangerous_code=True)

    res=agent.invoke(query)
    return res