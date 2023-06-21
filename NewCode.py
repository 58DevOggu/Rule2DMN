import spacy
import csv

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Define a function to process the training data
def process_training_data(text, entities):
    doc = nlp(text)
    
    # Initialize empty lists to store the detected entities and their types
    detected_entities = []
    detected_types = []
    
    for entity in entities:
        entity_text = entity[0]
        entity_type = entity[1]
        
        # Check if the entity is present in the processed document
        if entity_text.lower() in doc.text.lower():
            detected_entities.append(entity_text)
            detected_types.append(entity_type)
    
    return detected_entities, detected_types

# Define the path to the training dataset CSV file
training_file = 'training.csv'

# Initialize empty lists to store the processed training data
processed_texts = []
processed_entities = []
processed_types = []

# Read the training data from the CSV file
with open(training_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    
    for row in reader:
        text = row[0]
        entities = eval(row[1])  # Convert the entities string to a list of tuples
        
        detected_entities, detected_types = process_training_data(text, entities)
        
        processed_texts.append(text)
        processed_entities.append(detected_entities)
        processed_types.append(detected_types)

# Print the processed training data
for i in range(len(processed_texts)):
    print(f'Text: {processed_texts[i]}')
    print(f'Detected Entities: {processed_entities[i]}')
    print(f'Detected Types: {processed_types[i]}')
    print()

# Now you can use the processed training data to build your NLP model
# and train it to generate DMNs or perform other tasks as required.
