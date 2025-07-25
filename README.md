# [TITLE OF RESEARCH PAPER]

This repository contains all code, data, and prompts used for the research paper: **[_____]**.

## Project Overview

Technology companies play an increasingly visible role in political life. Through decisions around content moderation, platform access, and executive commentary, companies like Meta, Google, X/Twitter, and OpenAI have become central figures in public debates. These actions are often described as forms of **politicization** — where corporate decisions are interpreted as aligning with or influencing political agendas.

Reddit has emerged as a key platform for such discussions, offering a unique view into public perceptions of politicization across diverse communities. However, little is known about the structure, sentiment, and evolution of this discourse.

This project aims to fill that gap by analyzing Reddit posts about the perceived politicization of major technology companies. Using a mixed-methods approach that incorporates large language models (LLMs), unsupervised clustering, and temporal analysis, we address two main research questions:

- **RQ1**: What are the prevailing themes and sentiment trends in Reddit discussions about the perceived politicization of tech companies?
- **RQ2**: How have these discussions and sentiments shifted over time, particularly in relation to major political or platform-related events from 2020–2025?

## Repository Structure

The repository is organized as follows:

```
tech-politicization-discourse/
│
├── Dataset/
├── LLM_prompts/
├── RQ1_analysis/
├── RQ2_analysis/
├── Relevance_labelling/
├── Sentiment_labelling/
└── Validation_samples/
```
---

### Dataset/

- `Unlabelled_full_dataset.csv` — All Reddit posts extracted using keyword-based search across selected subreddits.
- `LLM_labelled_posts.xlsx` — Subset of posts labeled by the GPT-4o model for relevance and sentiment (with rationale included).

---

### LLM_prompts/

This folder contains the prompt templates used with GPT-4o for classification and labeling tasks:

- `BERTopic_LLM_cluster_labelling_prompt.txt` — Used to generate theme names for unsupervised clusters.
- `Political_subreddits_relevance_prompt.txt` — Prompt for determining whether a Reddit post is relevant to the topic of tech politicization.
- `Political_subreddits_sentiment_prompt.txt` — Prompt for labeling sentiment as *positive*, *neutral*, or *negative*.

---

### RQ1_analysis/

- `RQ1_analysis.ipynb` — Jupyter notebook containing all code and analysis related to **RQ1**: identifying themes and sentiment trends in discussions on tech politicization.

---

### RQ2_analysis/

- `RQ2_analysis.ipynb` — Jupyter notebook addressing **RQ2**: analyzing how themes and sentiment evolved over time, especially in relation to major events between 2020–2025.

---

### Relevance_labelling/

Contains Python scripts for labeling post **relevance** with LLMs:

- `Political_subreddit_LLM_relevance_labelling.py` — Classifies posts from political subreddits.
- `Tech_subreddit_LLM_relevance_labelling.py` — Classifies posts from tech-oriented subreddits.

---

### Sentiment_labelling/

Contains scripts for classifying **sentiment** of relevant posts:

- `Political_subreddit_LLM_sentiment_labelling.py` — For posts from political subreddits.
- `Tech_subreddit_LLM_sentiment_labelling.py` — For posts from tech subreddits.

---

### Validation_samples/

These files contain samples labeled by both authors to provide a 95% confidence interval validation for the LLM-generated relevance and sentiment classifications:

- `95CI_political_subreddit_relevance_validation.xlsx` — Relevance validation for political subreddits.
- `95CI_political_subreddit_sentiment_validation.xlsx` — Sentiment validation for political subreddits.
- `95CI_tech_subreddit_relevance_validation.xlsx` — Relevance validation for tech subreddits.
- `95CI_tech_subreddit_sentiment_validation.xlsx` — Sentiment validation for tech subreddits.

---

## Methods Summary

- Keyword-based extraction from 26 subreddits
- Relevance and sentiment classification via GPT-4o
- Thematic clustering using **BERTopic**
- Time-series and changepoint analysis using **PELT** and **ARIMA**
- Validation with human-labeled sample (95% CI)

---
