import numpy as np
import torch
# from plopy import list_files, HDF5Export, HDF5Dataset, process_vti_file
# from sipbuild.generator.parser.rules import p_preinit_code
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# # from nbc.training.postprocessing.lbm import Velocity
# from nbc.training.postprocessing.lbm import *
from training import NeuralBoundary

dtype = torch.float64
device = "cpu"
path = 'model_trained.pt'
model = torch.load(path, weights_only=False)
model.eval()
output_file = "model_exported.txt"

plusInput = True

with open(output_file, "w") as f:
    def initialize(model, num_layer):
        f.write("CudaDeviceFunction void EPressure_nbc() {\n")
        f.write("    \n")
        num_sequential = len(model.net)
        hidden_layer = 0
        for layer in range(num_sequential):
            if layer in [0]:
                if layer == 0:
                    nodes = model.net[layer].in_features
                    f.write(f"    float input[{nodes}] = {{0}};  // Initialize hidden layer nodes \n")
                continue
            if model.net[layer].__class__.__name__ == "Linear":
                hidden_layer += 1
                nodes = model.net[layer].in_features
                f.write(f"    float node{hidden_layer}[{nodes}] = {{0}};  // Initialize hidden layer nodes \n")
            if layer == num_sequential - 1:
                nodes = model.net[layer].out_features
                f.write(f"    float out[{nodes}] = {{0}};  // Initialize hidden layer nodes \n")
        f.write("    \n")
        f.write("    // Initialize 'input' array with function values\n")
        if model.net[0].in_features==6:
            f.write("    input[0] = f0;\n")
            f.write("    input[1] = f1;\n")
            f.write("    input[2] = f2;\n")
            f.write("    input[3] = f4;\n")
            f.write("    input[4] = f5;\n")
            f.write("    input[5] = f8;\n")
        elif model.net[0].in_features==9:
            f.write("    input[0] = f0(-1, 0);\n")
            f.write("    input[1] = f1(-1, 0);\n")
            f.write("    input[2] = f2(-1, 0);\n")
            f.write("    input[3] = f3(-1, 0);\n")
            f.write("    input[4] = f4(-1, 0);\n")
            f.write("    input[5] = f5(-1, 0);\n")
            f.write("    input[6] = f6(-1, 0);\n")
            f.write("    input[7] = f7(-1, 0);\n")
            f.write("    input[8] = f8(-1, 0);\n")
        elif model.net[0].in_features == 15:
            f.write("    input[0] = f0(-1, 0);\n")
            f.write("    input[1] = f1(-1, 0);\n")
            f.write("    input[2] = f2(-1, 0);\n")
            f.write("    input[3] = f3(-1, 0);\n")
            f.write("    input[4] = f4(-1, 0);\n")
            f.write("    input[5] = f5(-1, 0);\n")
            f.write("    input[6] = f6(-1, 0);\n")
            f.write("    input[7] = f7(-1, 0);\n")
            f.write("    input[8] = f8(-1, 0);\n")
            f.write("    input[9] = f0;\n")
            f.write("    input[10] = f1;\n")
            f.write("    input[11] = f2;\n")
            f.write("    input[12] = f4;\n")
            f.write("    input[13] = f5;\n")
            f.write("    input[14] = f8;\n")
        f.write("\n\n")


    def operation_inputlayer(layer=1, num_input=None, num_output=None):
        f.write(f"for (int i = 0; i < {num_output}; i++) {{\n")
        f.write(f"    for (int j = 0; j < {num_input}; j++) {{\n")
        f.write(f"        node{layer}[i] += input[j] * weight{layer}[i][j];  // Matrix multiplication with input\n")
        f.write(f"    }}\n")
        f.write(f"    node{layer}[i] += bias{layer}[i];  // Add bias\n")
        f.write(f"}}\n\n")


    def operation_outputlayer(layer=-1, num_input=None, num_output=None):
        f.write(f"for (int i = 0; i < {num_output}; i++) {{\n")
        f.write(f"    for (int j = 0; j < {num_input}; j++) {{\n")
        f.write(
            f"        out[i] += node{layer - 1}[j] * weight{layer}[i][j];  // Matrix multiplication with layer before\n")
        f.write(f"    }}\n")
        f.write(f"    out[i] += bias{layer}[i];  // Add bias\n")
        f.write(f"}}\n\n")


    def operation_hiddenlayer(layer=None, num_input=None, num_output=None):
        f.write(f"for (int i = 0; i < {num_output}; i++) {{\n")
        f.write(f"    for (int j = 0; j < {num_input}; j++) {{\n")
        f.write(
            f"        node{layer}[i] += node{layer - 1}[j] * weight{layer}[i][j];  // Matrix multiplication with layer before\n")
        f.write(f"    }}\n")
        f.write(f"    node{layer}[i] += bias{layer}[i];  // Add bias\n")
        f.write(f"}}\n\n")


    def operation_ReLU(layer=None, num_nodes=None):
        f.write(f"for (int i = 0; i < {num_nodes}; i++) {{\n")
        f.write(f"    if (node{layer}[i] < 0) {{\n")
        f.write(f"        node{layer}[i] = 0;  // Set negative values to 0\n")
        f.write(f"    }}\n")
        f.write(f"}}\n\n")


    def header(model):
        length_sequential = len(model.net)
        num_layer = 0
        for layer in range(length_sequential):
            if model.net[layer].__class__.__name__ == "Linear":
                num_layer += 1
                shape = model.net[layer].weight.data.shape
                f.write(f"float weight{num_layer}[{shape[0]}][{shape[1]}] = {{\n")
                weights = model.net[layer].weight.data.tolist()
                bias = model.net[layer].bias.data.tolist()
                for row in weights:
                    f.write("    { " + ", ".join(f"{value:.12f}" for value in row) + " },\n")
                f.write("};\n\n")
                f.write(f"float bias{num_layer}[{shape[0]}] = {{\n")
                f.write("    " + ", ".join(f"{value:.12f}" for value in bias) + "\n")
                f.write("};\n\n")


    def assignment(model, plusInput=False):
        if plusInput:
            f.write("f3 = out[0] + f3(-1, 0);\n")
            f.write("f6 = out[1] + f6(-1, 0);\n")
            f.write("f7 = out[2] + f7(-1, 0);\n")
        else:
            f.write("f3 = out[0];\n")
            f.write("f6 = out[1];\n")
            f.write("f7 = out[2];\n")
        f.write("\n\n")
        f.write("}")

    num_sequential = len(model.net)
    num_layer = 0
    for layer in range(num_sequential):
        if model.net[layer].__class__.__name__ == "Linear":
            num_layer += 1

    print(f"Number of layers: {num_layer}\n")
    print(f"Number of sequential modules: {num_sequential}\n\n")

    initialize(model, num_layer)  # Include the initialization function
    header(model)
    layer_i = 0
    for layer in range(num_sequential):
        if model.net[layer].__class__.__name__ == "Linear":
            layer_i += 1
            if layer_i == 1:
                operation_inputlayer(layer=1,
                                     num_input=model.net[layer].in_features,
                                     num_output=model.net[layer].out_features)
            elif layer_i == num_layer:
                operation_outputlayer(layer=num_layer,
                                      num_input=model.net[layer].in_features,
                                      num_output=model.net[layer].out_features)
            else:
                operation_hiddenlayer(layer=layer_i,
                                      num_input=model.net[layer].in_features,
                                      num_output=model.net[layer].out_features)
        else:
            operation_ReLU(layer=layer_i, num_nodes=model.net[layer - 1].out_features)

    assignment(model, plusInput)

print(f"Output has been written to {output_file}")
