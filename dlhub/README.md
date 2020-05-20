# Model Hosting on DLHub

These directories illustrate how to use our cloud-hosted versions of the ML
models for predicting G4MP2 formation energies.

The subdirectories are broken up by the type of model and each contain
a `publish_model.py` file that creates a DLHub servable for a certain
machine learning model.
The models themselves are stored in their own separate directory along with
an `about.yml` file that contains the metadata about the model.
`publish_model.py` file reads from the `about.yml` to create a full
metadata description and publishes the model files along with
a python API to invoke the model defined in `dlhub_sdk.py`
