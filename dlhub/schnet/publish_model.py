"""Command-line script for publishing SchNet models to DLHub"""

from dlhub_sdk.models.servables.python import PythonStaticMethodModel
from dlhub_sdk.utils.types import compose_argument_block
from dlhub_sdk.client import DLHubClient
from dlhub_app import evaluate_molecules
from argparse import ArgumentParser
from time import sleep, perf_counter
import numpy as np
import logging
import yaml
import json
import os

# Make a logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define test input data
mols = ['3\nH2 O1\nO -0.034360 0.977540 0.007602\nH 0.064766 0.020572 0.001535\nH 0.871790 1.300792 0.000693']
b3lyp = [-76.404702]

# Parse the user input
parser = ArgumentParser(description='''Post a SchNet-based delta-learning model to DLHub.

Requires that the model is in a directory with a YAML file containing metadata that
describes the provenance and purpose of the model. The model must be named "model.pkl.gz"
and the metadata named "about.yml"''')
parser.add_argument('--test', help='Just test out the submission and print the metadata', action='store_true')
parser.add_argument('directory', help='Model directory')
args = parser.parse_args()
logging.info(f'Starting publication from: {args.directory}')

# Check that the directory is formatted properly
model_dir = os.path.join(args.directory, 'model')
metadata_file = os.path.join(args.directory, 'about.yml')
if not (os.path.isdir(model_dir) and os.path.isfile(metadata_file)):
    raise ValueError('Directory must contain a "model" directory and "about.yml" with model and metadata"')
                     
# Load in the metadata
with open(metadata_file) as fp:
    metadata = yaml.load(fp, Loader=yaml.FullLoader)
logging.info(f'Read in metadata from: {metadata_file}')

# Write out the generic components of th emodel
model = PythonStaticMethodModel.from_function_pointer(evaluate_molecules)

#   Descriptions of the model interface
model.set_outputs('list', 'Estimate of G4MP2 atomization energy', item_type='float')

#   Provenance information for the model
model.add_related_identifier("10.1021/acs.jctc.8b00908", "DOI", "Requires")

#   DLHub search tools
model['datacite']['subjects'] = [{'subject': 'G4MP2 Delta Learning'}]
logging.info('Initialized model with generic metadata')

#   Computational environment
#     Add pointers to the correct model and save the fidelity
if os.path.lexists('model'):
    os.unlink('model')
os.symlink(model_dir, 'model')
with open('input_key.json', 'w') as fp:
    json.dump(metadata['input_key'], fp)

model.add_file('dlhub_app.py')
model.add_file(os.path.join('model', 'architecture.pth'))
model.add_file(os.path.join('model', 'best_model'))
model.add_file('train_dataset.pkl')
model.add_file('input_key.json')
model.add_directory('jcesr_ml', recursive=True, include='*.py')
model.parse_repo2docker_configuration()

# Add in the model-specific data

#   Model name
model.set_name(f'g4mp2_delta_schnet_{metadata["input_fidelity"].lower()}')
model.set_title(f'SchNet Model to Predict G4MP2 Atomization Energy from {metadata["input_fidelity"]} Total Energy')
logging.info(f'Defined model name: {model.name}')

#   Describe the function inputs
model.set_inputs(
    'tuple', 'Structures and B3LYP energies of moelcule sto be evaluated',
    element_types=[
        compose_argument_block('list', 'Structures of molecules in XYZ format. Structure should be relaxed with B3LYP/6-31G(2df,p)', item_type='string'),
        compose_argument_block('list', f'{metadata["input_fidelity"]} total energies of Energies of molecules in Ha', item_type='float')
    ])
model.set_unpack_inputs(True)

#   Model provenance information
model.set_authors(*zip(*metadata['authors']))
model.add_related_identifier(metadata['publication'], 'DOI', 'IsDescribedBy')
logging.info('Added model-specific metadata')

# If desired, print out the metadata
if args.test:
    logging.info(f'Metadata:\n{yaml.dump(model.to_dict(), indent=2)}')
    
    logging.info('Running function')
    local_result = evaluate_molecules(mols, b3lyp)
    logging.info(f'Success! Output: {local_result}')
    exit()

    
# Publish servable to DLHub

#   Make the client and submit task
client = DLHubClient(http_timeout=-1)
task_id = client.publish_servable(model)
logging.info(f'Started publication. Task ID: {task_id}')

#   Loop until publication completes
while client.get_task_status(task_id)['status'] != 'SUCCEEDED':
    sleep(30)
status = client.get_task_status(task_id)
logging.info('Finished publication')


# Test the servable

# Run it on DLHub
logging.info('Testing the DLHub servable')
dlhub_start = perf_counter()
dlhub_result = client.run(client.get_username() + '/' + model.name, 
                          (mols, b3lyp))
dlhub_runtime = perf_counter() - dlhub_start
logger.info(f'DLHub test completed. Runtime: {dlhub_runtime: .2f}')


# Run it on DLHub
logging.info('Testing the functional locally')
local_start = perf_counter()
dlhub_result = client.run(client.get_username() + '/' + model.name, 
                          (mols, b3lyp))
local_runtime = perf_counter() - dlhub_start
logger.info(f'Local test completed. Runtime: {dlhub_runtime: .2f}')

assert np.isclose(local_result, dlhub_result).all(), 'Results differ!'
logger.info('Complete.')
