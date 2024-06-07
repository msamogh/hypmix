from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate
from langsmith import Client
from dotenv import load_dotenv

load_dotenv(".env.secret")


# Constants
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 1.5
NUM_GENERATIONS_PER_SAMPLE = 1
SWEEP_NAME = "persistsim-sweep-1"
EXPERIMENT_PREFIX = "trial-1"
SPLIT = "calibration" # "evaluation"
PROMPT_NAME = "msamogh/persistsim-trial"

# Initialize client
client = Client()

# Load components
prompt = hub.pull(PROMPT_NAME)
chat_model = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
output_parser = StrOutputParser()

# Run evaluation
chain = prompt | chat_model | output_parser
results = evaluate(
    chain.invoke,
    data=client.list_examples(dataset_name=SWEEP_NAME, splits=[SPLIT]),
    experiment_prefix=EXPERIMENT_PREFIX,
    num_repetitions=NUM_GENERATIONS_PER_SAMPLE,
)
