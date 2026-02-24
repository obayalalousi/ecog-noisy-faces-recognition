# RECAP Project 
This repository contains the resources and code for the RECAP project, developed during the 3-week Neuromatch Academy 2025 program in Computational Neuroscience.  
**Title: "Effect of Visual Noise on High-Gamma ECoG Response Latency in the Fusiform Gyrus During Face Recognition."**  
The project is based on the electrocorticography (ECoG) dataset by Kai Miller on face and house image recognition tasks.  
This study investigates the relationship between the level of visual noise in face images and the latency of neural responses in high-gamma band power during correctly recognized face trials.

## Project Team
**Yulia Afletunova¹, Temitope Asama², Maxim Martaler³, Sinchana Vaasanthi⁴, Obay Alalousi⁵⁶**  
¹ Independent Researcher, Berlin, Germany.  
² Faculty of Basic Medical Sciences, University of Ibadan, Ibadan, Nigeria.  
³ System Neurobiology, University of Bremen, Bremen, Germany.  
⁴ Cognitive and Brain Sciences, Indian Institute of Technology, Gandhinagar, India.  
⁵ Neurology Department, AP‑HP Paris Public Hospitals, Paris, France.  
⁶ Faculty of Medicine, Sorbonne Université, Paris, France.  

## Project Context
This project was completed as part of the Neuromatch Academy (NMA) 2025 3-week course in Computational Neuroscience.

## Abstract
Please refer to the dedicated abstract file included in this repository.

## Dataset 
This project uses one of several publicly available electrocorticography (ECoG) datasets from the Kai J. Miller Library (2019).  
You can find the library description on [PubMed](https://pubmed.ncbi.nlm.nih.gov/31451738/).

The "houses and faces" dataset is one of them and was originally used in the several publications, including:
- Miller, Kai J., et al. "Face percept formation in human ventral temporal cortex." Journal of neurophysiology 118.5 (2017): 2614-2627.
- Miller, Kai J., et al. "The physiology of perception in human temporal lobe is specialized for contextual novelty." Journal of neurophysiology 114.1 (2015): 256-263.

It includes data from 7 epileptic patients with ECoG recordings. Each subject performed two experiments in which images of houses and faces were presented on a screen. For more details about the two experiments, please refer to the main Jupyter notebook in this repository and to the original publications.
The dataset used here was curated by Neuromatch Academy ([project page](https://compneuro.neuromatch.io/projects/ECoG/README.html)) and downloaded from the Neuromatch academy [Open Science Framework (OSF)](https://osf.io/argh7).  
It is automatically downloaded when running the notebook.

Original raw data can also be accessed directly from Stanford’s data repository:  
https://exhibits.stanford.edu/data/catalog/zk881ps0522 

## Repository Contents
- `main.ipynb`: Main Jupyter notebook containing the code and analysis.
- `utils.py`: Python script with helper functions. They were designed to be reusable in other analyses. Feel free to reuse or modify them for your own work.
- `Presentation.pdf`: Presentation slides summarizing the project.
- `Abstract.md`: Project abstract.
- `requirements.txt`: List of Python dependencies required to run the notebook.
- `LICENSE`: License file (MIT).

## Installation
To run this project, you may need to install the required dependencies listed in requirements.txt.

## AI Disclosure
A generative language model was used to assist in drafting the docstrings.  

## Citation
If you build upon this work in your own research or project, please consider citing this repository:  

    Y. Afletunova, T. Asama, M. Martaler, S. Vaasanthi, O. Alalousi (2025). Effect of Visual Noise on High-Gamma ECoG Response Latency in the Fusiform Gyrus During Face Recognition. Project in Cognitive Computational Neuroscience. GitHub (https://github.com/obayalalousi/ecog-noisy-faces-recognition/).

## License
This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute the code with proper attribution.

## Acknowledgments
We thank Neuromatch Academy, as well as the Teaching Assistant M. Ritobrata Ghosh, Project Assistant M. Ali Mohammadnezhad, and Project Mentor M. Sankaraleengam Alagapan for their guidance and support throughout the program.
