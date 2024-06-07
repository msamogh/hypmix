from pprint import pprint

from langchain import hub
from langsmith import Client

from dotenv import load_dotenv

load_dotenv(".env.secret")


# Constants
SWEEP_NAME = "persistsim-sweep-1"


client = Client()


def filter_by_criteria(criteria):
    return client.list_examples(dataset_name=SWEEP_NAME, metadata=criteria)


if __name__ == "__main__":
    pprint(str(next(filter_by_criteria({"split": "calibration"}))))
