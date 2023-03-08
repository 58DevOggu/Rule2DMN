import json
import requests
import spacy

# Load the spaCy model for English language
nlp = spacy.load("en_core_web_sm")

# Define the input rule text
rule_text = input("Enter the rule text: ")

# Use spaCy to extract the input variables and values from the rule text
doc = nlp(rule_text)

credit_score = None
marital_status = None

for token in doc:
    if token.text == "Credit" and token.nbor(1).text == "score" and token.nbor(2).text == "is":
        credit_score = int(token.nbor(3).text)
    elif token.text == "married" and token.dep_ == "attr":
        marital_status = token.text

# Define the DMN engine URL
dmn_engine_url = "https://dmn.camunda.cloud/api/v1/evaluate"

# Define the DMN model
dmn_model = {
    "model": {
        "inputs": {},
        "outputs": {},
        "rules": [],
    }
}

# Add the input variables to the DMN model
if credit_score is not None:
    dmn_model["model"]["inputs"]["credit_score"] = {
        "type": "number"
    }
    dmn_model["model"]["rules"].append({
        "inputEntries": [
            {
                "text": "credit_score",
                "inputName": "credit_score",
                "value": str(credit_score)
            }
        ],
        "outputEntries": [
            {
                "text": "loan_approval",
                "outputName": "loan_approval",
                "value": "true"
            }
        ]
    })

if marital_status is not None:
    dmn_model["model"]["inputs"]["marital_status"] = {
        "type": "string"
    }
    dmn_model["model"]["rules"].append({
        "inputEntries": [
            {
                "text": "marital_status",
                "inputName": "marital_status",
                "value": marital_status
            }
        ],
        "outputEntries": [
            {
                "text": "loan_approval",
                "outputName": "loan_approval",
                "value": "true"
            }
        ]
    })

# Convert the DMN model to JSON format
dmn_model_json = json.dumps(dmn_model)

# Define the DMN evaluation payload
dmn_payload = {
    "variables": {
        "input": dmn_model_json
    }
}

# Evaluate the DMN model using the DMN engine API
response = requests.post(dmn_engine_url, json=dmn_payload)

# Extract the DMN evaluation result
dmn_result = response.json()
loan_approval = dmn_result["outputs"]["loan_approval"]["value"]

# Print the loan approval decision
if loan_approval == "true":
    print("Loan approved!")
else:
    print("Loan not approved.")
