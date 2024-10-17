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


If you use this repository, please consider citing our paper. The BibTex for our paper is:
```
@article{mannekote2024can,
  title={Can LLMs Reliably Simulate Human Learner Actions? A Simulation Authoring Framework for Open-Ended Learning Environments},
  author={Mannekote, Amogh and Davies, Adam and Kang, Jina and Boyer, Kristy Elizabeth},
  journal={arXiv preprint arXiv:2410.02110},
  year={2024}
}
```
