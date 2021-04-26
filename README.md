# Mordecai v3


## Setup

To install the libraries required for Mordecai, `pip install` the list of required libraries:

```
pip install -r requirements.txt
```

Then run the following command to download the NLP model used to identify place names:

```
python -m spacy download en_core_web_trf
```

The event-location linking step requires other models to be downloaded from https://huggingface.co/. These will be automatically downloaded the first time the program is run (if it's run on an internet-connected machine).

## Usage

To batch-process a JSONL file of ICEWS events, run the following command:

```
python batch_process.py storiesWithEvents-3.json
```
