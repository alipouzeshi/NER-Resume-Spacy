Resume Named Entity Recognition (NER) with spaCy
This repository contains a complete pipeline for training a custom Named Entity Recognition (NER) model to extract key information from resumes. The model is built using spaCy and trained on a manually annotated dataset of resumes. It can identify entities such as name, location, email, skills, education, work experience, and more.

📌 Overview
Automated information extraction from resumes is a critical task for HR tech, job portals, and applicant tracking systems. This project implements a spaCy NER model that can parse unstructured resume text and output structured entities. The entire workflow includes:

Data loading and conversion from DataTurks JSON format

Cleaning and handling overlapping entity spans

Training a blank spaCy English model with a custom NER pipeline

Evaluation at both token level (BIO tagging) and entity level (exact span match)

The final model is saved and can be reused for inference on new resumes.

📊 Dataset
The dataset used is Entity Recognition in Resumes.json – a collection of resumes annotated in DataTurks format. Each entry contains:

content: raw resume text (with newlines replaced by spaces)

annotation: list of annotated entities with start/end offsets and labels

Entity Types
The following entity labels are present in the dataset:

Entity Label	Description
Name	Candidate’s full name
Email Address	Email (often from Indeed or other domains)
Location	City, state, or region
Degree	Academic degree (e.g., B.E., B.Tech)
College Name	Name of educational institution
Graduation Year	Year of graduation (4‑digit)
Skills	Technical or soft skills
Designation	Job title or role
Companies worked at	Past employer names
Years of Experience	Total years of professional experience
Note: The dataset contains some overlapping and misaligned entity spans. Preprocessing steps are included to clean these issues.

⚙️ Preprocessing
The notebook applies several preprocessing functions to prepare the data for spaCy training:

1. convert_dataturks_to_spacy(dataturks_JSON_FilePath)
Reads the DataTurks JSON file line by line

Extracts text and annotations

Adjusts entity offsets to ignore leading/trailing whitespace

Returns a list of (text, {"entities": [(start, end, label), ...]}) suitable for spaCy

2. trim_entity_spans(data)
Removes leading/trailing whitespace inside entity spans

Updates start/end offsets accordingly

3. clean_entities(training_data)
Removes overlapping entities when a shorter entity is fully contained inside a longer one

Keeps the longest, most informative entity

4. train_test_split(data, test_size, random_state)
Shuffles and splits data into training (90%) and test (10%) sets

Fixed random seed (42) for reproducibility

🧠 Model Training
The training is performed using spaCy’s built-in NER training loop:

A blank English model (spacy.blank('en')) is created

An NER pipeline component is added

Entity labels are inferred from the training data and added to the pipe

Training runs for 100 iterations with:

drop = 0.2 for regularisation

Stochastic gradient descent (sgd=optimizer)

All non‑NER pipeline components are disabled during training for efficiency

The training loop prints the iteration number and the NER loss at each step.
Loss decreases from ~15,500 to ~640 over 100 iterations.

python
nlp_model = train_spacy(train_data, n_iter=100)
nlp_model.to_disk("ner_resume_model")
📈 Evaluation
Two complementary evaluation strategies are implemented:

A. Token‑level (BIO tagging)
Tokens are aligned with ground truth and predicted entities using char_span

Tags follow the BIO scheme: B-ENTITY (begin), I-ENTITY (inside), O (outside)

The seqeval library computes precision, recall, and f1‑score per entity type

B. Entity‑level (exact span match)
Entities are compared as (start_char, end_char, label) tuples

Counts correct, predicted, and gold per label

Micro and macro averages are reported

Both evaluation methods are applied to the held‑out test set (10% of data).

📊 Results
Entity‑level Exact Match (sample from notebook)
Entity Label	Precision	Recall	F1‑Score
Name	0.955	0.913	0.933
Degree	0.808	0.750	0.778
Email Address	0.667	0.667	0.667
College Name	0.643	0.500	0.563
Location	0.630	0.486	0.548
Designation	0.513	0.417	0.460
Companies worked at	0.578	0.361	0.444
Skills	0.478	0.333	0.393
Graduation Year	0.261	0.273	0.267
Years of Experience	0.000	0.000	0.000
Observation: The model performs best on Name and Degree, while Years of Experience and Graduation Year are harder due to sparse and varied formats. Overlapping spans and noisy annotations also affect performance on Skills and Companies worked at.

🚀 How to Use
1. Clone the repository and install dependencies
bash
git clone https://github.com/yourusername/resume-ner-spacy.git
cd resume-ner-spacy
pip install -r requirements.txt
2. Train the model from scratch
Run the Jupyter notebook Resume_NER.ipynb step by step. The final model is saved as ner_resume_model/.

3. Load the trained model for inference
python
import spacy

nlp = spacy.load("ner_resume_model")
resume_text = """
John Doe, Software Engineer with 5 years of experience in Python and AWS. 
Graduated from Stanford University in 2018. Email: john.doe@example.com
"""
doc = nlp(resume_text)
for ent in doc.ents:
    print(f"{ent.label_}: {ent.text}")
📁 Repository Structure
text
.
├── Resume_NER.ipynb              # Main training & evaluation notebook
├── Entity Recognition in Resumes.json   # Annotated dataset (not included here)
├── ner_resume_model/             # Saved spaCy model (generated after training)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
📦 Dependencies
Python 3.7+

spaCy (>=3.0)

seqeval

numpy, random, re, json (standard library)

Install with:

bash
pip install spacy seqeval
python -m spacy download en_core_web_sm   # optional, for additional pipelines
📝 Future Improvements
Add more training data (especially for Years of Experience and Graduation Year)

Use a pre‑trained transformer model (e.g., en_core_web_trf) as base

Implement data augmentation (e.g., synonym replacement, format variations)

Add a web API (Flask/FastAPI) for serving predictions

🤝 Acknowledgements
Dataset originally from DataTurks (public resume NER dataset)

Built with spaCy – Industrial‑strength NLP
