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

import csv

def read_input_data(file_path):
    input_data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            text = row[0]
            entities = []
            entity_string = row[1][1:-1]  # Remove the outer parentheses
            entity_pairs = entity_string.split('), (')
            for entity_pair in entity_pairs:
                entity, attribute = entity_pair.split(', ')
                entity = entity.strip()[1:-1]  # Remove the outer quotes
                attribute = attribute.strip()[1:-1]  # Remove the outer quotes
                entities.append((entity, attribute))
            input_data.append((text, entities))
    return input_data

# Example usage:
input_data = read_input_data('input_data.csv')
print(input_data)
def generate_dmn_from_input(input_data):
    dmn_template = '''<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/DMN/20180521/MODEL/" xmlns:dmndi="http://www.omg.org/spec/DMN/20180521/DMNDI/" xmlns:dc="http://www.omg.org/spec/DMN/20180521/DC/" xmlns:di="http://www.omg.org/spec/DMN/20180521/DI/" xmlns:feel="http://www.omg.org/spec/DMN/20180521/FEEL/" xmlns:dmn="http://www.omg.org/spec/DMN/20180521/DMN/" id="definitions" name="definitions" namespace="http://camunda.org/schema/1.0/dmn">
  <decision id="decision" name="Decision" dmn:decisionLogic="decisionTable">
    <decisionTable id="decisionTable" hitPolicy="UNIQUE">
      <!-- Input entries -->
      {input_entries}
      <!-- Output entries -->
      {output_entries}
    </decisionTable>
  </decision>
</definitions>
'''

    input_entry_template = '''
      <inputEntry id="{input_id}">
        <text>{input_text}</text>
      </inputEntry>
    '''

    output_entry_template = '''
      <outputEntry id="{output_id}">
        <text>{output_text}</text>
      </outputEntry>
    '''

    input_entries = ''
    output_entries = ''

    for idx, input_tuple in enumerate(input_data):
        text, entities = input_tuple
        input_id = f'input{idx + 1}'
        input_text = ' '.join([entity for entity, _ in entities])
        output_id = f'output{idx + 1}'
        output_text = ' '.join([entity for entity, entity_type in entities if entity_type == 'OUTPUT'])

        input_entry = input_entry_template.format(input_id=input_id, input_text=input_text)
        output_entry = output_entry_template.format(output_id=output_id, output_text=output_text)

        input_entries += input_entry
        output_entries += output_entry

    dmn_content = dmn_template.format(input_entries=input_entries, output_entries=output_entries)
    return dmn_content

# Example usage:
dmn_content = generate_dmn_from_input(input_data)
print(dmn_content)



