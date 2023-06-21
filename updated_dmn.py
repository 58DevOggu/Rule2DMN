import spacy
import pandas as pd

# Step 1: Prepare the training data in Training.csv

# Step 2: Install spaCy and download language model
!pip install spacy
!python -m spacy download en_core_web_sm

# Step 3: Train the NER model
nlp = spacy.load('en_core_web_sm')

# Load and preprocess the training data
training_data = pd.read_csv('Training.csv')

TRAIN_DATA = []
for idx, row in training_data.iterrows():
    text = row['text']
    entities = [(start, end, label) for start, end, label in row['entities']]
    TRAIN_DATA.append((text, {"entities": entities}))

# Train the NER model
nlp.disable_pipes('tagger', 'parser')
ner = nlp.get_pipe("ner")
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

n_iter = 10
for itn in range(n_iter):
    losses = {}
    for text, annotations in TRAIN_DATA:
        nlp.update([text], [annotations], losses=losses)
    print(f"Iteration {itn}: Losses - {losses}")

# Step 4: Extract entity, attribute, input, and output information
def extract_information(text):
    doc = nlp(text)
    entities = []
    attributes = []
    inputs = []
    outputs = []
    for ent in doc.ents:
        if ent.label_ == "ENTITY":
            entities.append(ent.text)
        elif ent.label_ == "ATTRIBUTE":
            attributes.append(ent.text)
        elif ent.label_ == "INPUT":
            inputs.append(ent.text)
        elif ent.label_ == "OUTPUT":
            outputs.append(ent.text)
    return entities, attributes, inputs, outputs

# Step 5: Generate DMN decision table
def generate_dmn(entities, attributes, inputs, outputs):
    # Define DMN template and fill in the placeholders with the extracted information
    dmn_template = f'''
    <definitions>
        <decision>
            <variable>{", ".join(attributes)}</variable>
            <informationRequirement>
                <requiredInput>{", ".join(inputs)}</requiredInput>
            </informationRequirement>
            <decisionTable>
                <input>{", ".join(inputs)}</input>
                <output>{", ".join(outputs)}</output>
            </decisionTable>
        </decision>
    </definitions>
    '''
    return dmn_template

# Example usage
text = "If a Customer's Credit Score is > 700 then Approve Loan"
entities, attributes, inputs, outputs = extract_information(text)
dmn = generate_dmn(entities, attributes, inputs, outputs)

print(dmn)
