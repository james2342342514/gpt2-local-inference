# GPT-2 Local Inference

This project demonstrates that Large Language Models (LLMs) can run on consumer-grade hardware, contrary to the belief that they require supercomputers or cloud infrastructure. This project uses the GPT-2 medium model with PyTorch and the Transformers library.


## Project Description

This repository contains Python scripts that allow you to run pre-trained language models locally on your personal computer. It uses the Hugging Face Transformers library to load and run models like GPT-2, showcasing that modern consumer hardware is capable of running inference on these AI models.

## Hardware Specifications

This project runs successfully on the following consumer-grade hardware:

- CPU: AMD Ryzen 5 5600X
- GPU: NVIDIA RTX 2070 Super
- RAM: 16GB DDR4 2400

## Requirements 

- Python 3.8+
- PyTorch
- Transformers library
- CUDA-capable GPU (optional, for faster inference)

These specifications demonstrate that high-end supercomputers are not necessary for running inference on language models like GPT-2.

## Key Features

- Local execution of LLMs without cloud dependencies
- Utilizes consumer-grade GPU for accelerated inference
- Demonstrates the accessibility of AI technology
- Provides a starting point for personal AI experiments and learning

## Why This Matters

Many people believe that advanced AI models can only run on supercomputers or in the cloud. This project proves otherwise, showing that with the right setup, anyone with a decent gaming PC or workstation can experiment with and utilize these powerful language models.

By running these models locally, we can:
1. Maintain privacy by keeping data on our own machines
2. Reduce latency compared to cloud-based solutions
3. Learn about and experiment with AI technology hands-on
4. Demonstrate the rapid advancement and democratization of AI technology

Whether you're a student, hobbyist, or professional developer, this project aims to demystify LLMs and make them more accessible to everyone.

### Windows 11: Missing DLL Error

If you encounter an error related to missing DLL files (specifically `fbgemm.dll`) on Windows 11, follow these steps to resolve the issue:

1. Download and run the Visual Studio Installer.
2. Select "Modify" if you have Visual Studio installed, or proceed with a new installation.
3. Go to the "Individual components" tab.
4. Search for "C++ 2022" in the search box.
5. Check the box next to "MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)".
6. Click "Modify" or "Install" to proceed with the installation.
7. Restart your computer after the installation is complete.

This installs the necessary C++ build tools and runtime libraries that PyTorch depends on.

## Usage

- Run the script using: python run_llm.py
- The script will output the device being used (CPU or CUDA), generate text based on a default prompt, and display the results.
- You can modify the `prompt` variable in the script to generate text based on different inputs.

