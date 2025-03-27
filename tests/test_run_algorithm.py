"""
Test suite for the run_algorithm function, which is tested both server-side and client-side.
"""

import sys
import os
import subprocess
import time

import numpy as np
from imaging_server_kit.client import Client

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import Server


def run_algorithm_server_side():
    # Create a Server instance
    server = Server()

    # Get the algorithm parameters
    algo_params_schema = server.parameters_model.model_json_schema()
    required_params = algo_params_schema.get("required")
    algo_params = algo_params_schema.get("properties")

    # Get default parameter values for all non-requried parameters (which do not have defaults)
    default_params = {}
    for param, param_values in algo_params.items():
        if param not in required_params:
            default_params[param] = param_values.get("default")

    # Add values to the required parameters, for example an input image
    sample_image = server.load_sample_images()[0]  # First sample image
    default_params["image"] = sample_image

    # All parameters must have values set to run the algorithm
    params_missing = set(default_params.keys()) - set(algo_params.keys())
    assert (
        len(params_missing) == 0
    ), f"Values are missing for required parameters: {params_missing}"

    # Run the algorithm
    algo_output = server.run_algorithm(**default_params)

    # Examine the output (add relevant assert statements to test the algorithm)
    for data, data_params, data_type in algo_output:
        if data_type == "image":
            assert isinstance(
                data, np.ndarray
            ), "Algorithm did not output a Numpy array."
        elif data_type == "mask":
            assert isinstance(
                data, np.ndarray
            ), "Algorithm did not output a Numpy array."

    return algo_output


def run_algorithm_client_side():
    # Start the FastAPI server using uvicorn
    server_process = subprocess.Popen(
        ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]
    )
    time.sleep(5)  # Wait for the server to start

    try:
        # Connect to the server
        client = Client("http://localhost:8000")

        # A single algorithm should be available
        assert (
            len(client.algorithms) > 0
        ), f"No algorithm available: {client.algorithms}"
        assert (
            len(client.algorithms) < 2
        ), f"More than one algorithm available: {client.algorithms}"

        # Get the algorithm parameters
        algo_params_schema = client.get_algorithm_parameters()
        required_params = algo_params_schema.get("required")
        algo_params = algo_params_schema.get("properties")

        # Get default parameter values for all non-requried parameters (which do not have defaults)
        default_params = {}
        for param, param_values in algo_params.items():
            if param not in required_params:
                default_params[param] = param_values.get("default")

        # Add values to the required parameters, for example an input image
        default_params["image"] = client.get_sample_images(
            first_only=True
        )  # First sample image

        # All parameters must have values set to run the algorithm
        default_params_set = set(default_params.keys())
        algo_params_set = set(algo_params.keys())
        params_missing = algo_params_set.difference(default_params_set)

        assert (
            len(params_missing) == 0
        ), f"Values are missing for required parameters: {params_missing}"

        # Run the algorithm
        algo_output = client.run_algorithm(**default_params)
    finally:
        # Shut down the server
        server_process.terminate()
        server_process.wait()

    return algo_output


def test_compare_algo_outputs():
    server_output = run_algorithm_server_side()
    client_output = run_algorithm_client_side()

    for server_data_tuple, client_data_tuple in zip(server_output, client_output):
        assert (
            server_data_tuple[2] == client_data_tuple[2]
        ), "Server and client output data types are different."
        assert np.allclose(
            server_data_tuple[0], client_data_tuple[0]
        ), "Server and client algorithm outputs do not match."
