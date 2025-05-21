<p align="center">
<img src="climatecheck-logo.gif" alt="drawing" width="300"/>
  <br>
</p>

## ClimateCheck: Fact-Checking Social Media Posts About Climate Change Against Scholarly Articles

This repository contains the code for creating the ClimatCheck dataset, which is used for ClimateCheck shared task hosted at the 5th Scholarly Document Processing Workshop @ ACL in Vienna, Austria. More information can be found at:  [https://sdproc.org/2025/climatecheck.html](https://sdproc.org/2025/climatecheck.html)

### Dataset Creation 

Two datastets were created, both are available at 🤗 HuggingFace: 
* ClimateCheck dataset (training + testing): [https://huggingface.co/datasets/rabuahmad/climatecheck](https://huggingface.co/datasets/rabuahmad/climatecheck)
* Publications Corpus:  [https://huggingface.co/datasets/rabuahmad/climatecheck_publications_corpus](https://huggingface.co/datasets/rabuahmad/climatecheck_publications_corpus)

#### Collection of Claims

The claims used for this dataset were gathered from the following existing resources: ClimaConvo, DEBAGREEMENT, Climate-Fever, MultiFC, and ClimateFeedback. Some of which are extracted from social media (Twitter/X and Reddit) and some were created synthetically from news and media outlets using text style transfer techniques to resemble tweets. All claims underwent a process of scientific check-worthiness detection and are formed as atomic claims (i.e. containing only one core claim).

#### Collection of Publications

To retrieve relevant abstracts, a corpus of publications was gathered from OpenAlex and S2ORC, containining 394,269 abstracts. 

#### Annotation Process

The data was annotated by five graduate students in the Climate and Environmental Sciences. Using a TREC-like pooling approach, we retrieved the top 20 abstracts for each claim using BM25 followed by a neural cross-encoder trained on the MSMARCO data. Then we used 6 state-of-the-art models to classify claim-abstract pairs. If a pair resulted in at least 3 evidentiary predictions, it was added to the annotation corpus. Each claim-abstract pair was annotated by two students, and resolved by a curator in cases of disagreements.

#### Repository Structure

```
    ├── src               
    │   ├── claims_prep    # scripts for extracting and preprocessing claims       
    │   ├── publications_prep     # scripts for gathering climate-related publications and post-processing
    │   ├── linking          # scripts for linking claims to relevant publications, creating the annotation corpus

```

### Shared Task

The task was hosted on [Codabench](https://www.codabench.org/competitions/6639/) and contained the following subtasks: 

Given a claim in English extracted from a social media platform about climate change: 
1. **Subtask I:** Find all relevant publications related to it from a pre-determined corpus of climate change research publications. 
2. **Subtask II:** For each of those, predict whether the publication supports, refutes, or does not have enough information about the claim.\
The predictions for each claim should be a list of related articles and their labels. 

