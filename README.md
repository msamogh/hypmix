# Classroom of LLMs

## Setup

### Set up Python dependencies
1. Create a virtualenv with Python 3.11
2. Install requirements
```bash
pip install -r requirements.txt
```

### Set up API access tokens
1. In `.env`, change `LANGCHAIN_PROJECT` to the ID of the LangSmith project you are using in your account.
2. Rename `.env.secret.template` to `.env.secret`. Fill in the values for each of the API keys for OpenAI and LangSmith.


## Running Experiments

TLDR: run `src/main.py` with different args for the specific experiment in question, and modify prompts as either: 
- modify template prompts in the associated LangSmith config (e.g., [here](https://smith.langchain.com/hub/amogh-ld/sl-calibration-1?organizationId=7a406b1a-9843-5799-b8d1-dfbf5b4154d1) > `ChatPromptTemplate`)
- modify specific components of prompts (slotted into LangSmith templates) by editing associated dataclasses, etc. in `action_spaces.py`, `state_spaces.py`, etc.
