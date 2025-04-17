AI-Powered Puppy Image Generator



Overview

Welcome to the AI-Powered Puppy Image Generator App, a cutting-edge Flask application that generates stunning hybrid puppy images using artificial intelligence. With two intuitive modes—Text Mode for blending breeds with custom ratios and Image Mode for breed detection from uploaded photos—this project harnesses Stable Diffusion on GPUs to deliver high-quality results, saved to /tmp and shared via ngrok.

Features





Text Mode: Seamlessly blend breed names with ratios (25%/75%, 50%/50%, 75%/25%) using detailed breed traits.



Image Mode: Leverage a Vision Transformer (ViT) model with a 0.05 confidence threshold to identify breeds from uploads.



AI-Powered Rendering: Utilize Stable Diffusion on GPUs for lifelike puppy images.



Reliability: Built-in retry logic ensures successful generation.



Accessibility: Share results effortlessly via ngrok tunneling.

Tech Stack





Backend: Flask, Python



AI: Stable Diffusion, Transformers, PyTorch



Breed Detection: ViT Model



Deployment: Ngrok



Storage: Temporary file system (/tmp)
