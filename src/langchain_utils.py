from dotenv import load_dotenv

from langsmith import Client

load_dotenv(".env.secret")
load_dotenv(".env")


def get_next_dataset_name():
    client = Client()
    datasets = list(client.list_datasets())
    name = f"persistsim-experiment-{len(datasets) + 10}"
    return name
