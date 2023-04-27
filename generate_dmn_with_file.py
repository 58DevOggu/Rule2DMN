import pandas as pd
import re
import spacy

# Read in the input file as a dataframe
input_df = pd.read_csv('input_file.csv')

# Clean and normalize the text
input_df['rule_text'] = input_df['rule_text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s\>\=\<\!]+', '', x.lower().strip()))

# Extract relevant metadata (e.g. entities and attributes)
input_df['entity'] = input_df['rule_text'].apply(lambda x: re.findall('[a-zA-Z]+', x)[0])
input_df['attribute'] = input_df['rule_text'].apply(lambda x: re.findall('[a-zA-Z]+', x)[1])

# Load the pre-trained NLP model
nlp = spacy.load('en_core_web_sm')

# Define a function to extract key concepts and entities
def extract_concepts(text):
    doc = nlp(text)
    concepts = []
    for ent in doc.ents:
        concepts.append({
            'entity': ent.label_,
            'value': ent.text,
            'start': ent.start_char,
            'end': ent.end_char
        })
    return concepts

# Apply the function to the preprocessed text data
input_df['concepts'] = input_df['rule_text'].apply(extract_concepts)

# Define a set of mapping rules to map concepts to DMN elements
mapping_rules = {
    'entity': {
        'customer': 'CustomerData',
        'loan': 'LoanData'
    },
    'attribute': {
        'credit score': 'CreditScore',
        'income': 'Income',
        'loan amount': 'LoanAmount',
        'loan term': 'LoanTerm',
        'marital status': 'MaritalStatus'
    },
    'value': {
        'approved': 'Approved',
        'rejected': 'Rejected',
        'pending': 'Pending'
    }
}

# Define a function to map concepts to DMN elements
def map_concepts_to_dmn(concepts):
    dmn_elements = []
    for concept in concepts:
        dmn_element = mapping_rules[concept['entity']][concept['value']] + '.' + mapping_rules[concept['entity']][concept['attribute']]
        dmn_elements.append(dmn_element)
    return dmn_elements

# Apply the function to the extracted concepts
input_df['dmn_elements'] = input_df['concepts'].apply(map_concepts_to_dmn)

def generate_dmn_include_dt(rule_text: str, input_variable: str, input_value: str, output_variable: str, output_value: str) -> str:
    inputData = f'<inputData id="{input_variable}" name="{input_variable}"><variable id="{input_variable}" name="{input_variable}" typeRef="number"/></inputData>'
    outputData = f'<outputData id="{output_variable}" name="{output_variable}"><variable id="{output_variable}" name="{output_variable}" typeRef="number"/></outputData>'
    
    decisionTable = f'<decisionTable id="decision" hitPolicy="UNIQUE" inputExpression="{input_variable}" outputExpression="{output_variable}">{rule_text}</decisionTable>'
    
    dmn_str = f'''<?xml version="1.0" encoding="UTF-8"?>
    <definitions xmlns="http://www.omg.org/spec/DMN/20151101/dmn.xsd" id="definitions" name="definitions">
    <decision id="decision" name="decision" expressionLanguage="http://www.omg.org/spec/FEEL/20140401">{inputData}{outputData}{decisionTable}</decision>
    </definitions>
    '''
    return dmn_str


# Define a function to generate DMNs from the mapped concepts
def generate_dmn(dmn_elements):
    dmn = '''
    <?xml version="1.0" encoding="UTF-8"?>
    <definitions xmlns="http://www.omg.org/spec/DMN/20151101/dmn.xsd"
        xmlns:dmndi="http://www.omg.org/spec/DMN/20151101/dmn-di.xsd"
        xmlns:di="http://www.omg.org/spec/DMN/20180521/DI/"
        xmlns:dc="http://www.omg.org/spec/DMN/20180521/DC/"
        id="definition"
        name="DMN Definition">
        <decision id="decision" name="Decision Name">
            <variable id="variable" name="Variable Name" typeRef="string" />
            <informationRequirement id="information_requirement">
                <requiredInput href="#input_data" />
            </informationRequirement>
            <decisionTable id="decision_table" hitPolicy="UNIQUE">
                <input id="input" label="Input Label" inputExpression="{{input_data}}" />
                <output id="output" label="Output Label" typeRef="string" />
                <rule id="rule" name="Rule Name">
                    <inputEntry id="input_entry">{{input_value}}</inputEntry>
                    <outputEntry id="output_entry">{{output_value}}</outputEntry>
                </rule>
            </decisionTable>
        </decision>
        <inputData id="input_data" name="Input Data Name" />
    </definitions>
    '''

    # Replace placeholders in the DMN template with the mapped concepts
    input_data = dmn_elements[0]
    input_value = dmn_elements[1] + '.' + dmn_elements[2]
    output_value = dmn_elements[3]

    dmn = dmn.replace('{{input_data}}', input_data)
    dmn = dmn.replace('{{input_value}}', input_value)
    dmn = dmn.replace('{{output_value}}', output_value)

    return dmn

# Apply the function to the mapped concepts
input_df['dmn'] = input_df['dmn_elements'].apply(generate_dmn)

# Define a function to write DMNs to files
def write_dmn_file(row):
    filename = row['rule_id'] + '.dmn'
    with open(filename, 'w') as f:
        f.write(row['dmn'])

# Apply the function to each row in the input dataframe
input_df.apply(write_dmn_file, axis=1)
