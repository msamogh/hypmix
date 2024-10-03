# HypMix

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


