# Project 3: Automated News Categorizer (LDA Topic Modeling) 🗞️

## Overview
This is the third project in my 6-part Natural Language Processing (NLP) series. Moving beyond supervised classification, this project focuses on **Unsupervised Learning** and **Topic Modeling**. 

Using the BBC News Archive, I built a pipeline that ingests raw text and uses **Latent Dirichlet Allocation (LDA)** to mathematically discover hidden semantic structures and group the news into natural clusters—without providing any predefined labels to the model.

## Dataset
* **Source:** [BBC News Archive (Kaggle)](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive)
* **Size:** 2,225 articles
* **Format:** Raw text articles originally labeled across 5 categories (Business, Entertainment, Politics, Sport, Tech).

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **NLP Processing:** NLTK (WordNetLemmatizer, Stopwords), Regular Expressions (Regex)
* **Machine Learning:** Scikit-learn (`CountVectorizer`, `LatentDirichletAllocation`)

## The Workflow

### 1. Deep Text Preprocessing
Preprocessing dictates the success of unsupervised learning. The text was rigorously cleaned to ensure the LDA model clustered meaningful topics rather than common grammar:
* Stripped non-alphabetical characters using Regex.
* Converted all text to lowercase.
* Removed English stopwords using NLTK.
* Applied **Lemmatization** via `WordNetLemmatizer` to reduce words to their base root (e.g., "running" → "run").

### 2. Feature Extraction
* Utilized `CountVectorizer` to convert the cleaned text documents into a matrix of token counts.
* **Hyperparameter Tuning:** Set `max_df=0.95` (ignoring words appearing in >95% of documents) and `min_df=2` (ignoring highly obscure words) to heavily filter out noise and improve topic clarity.

### 3. Topic Modeling (LDA)
* Initialized the `LatentDirichletAllocation` model with `n_components=5`.
* The model successfully grouped the corpus into 5 distinct topics based on word co-occurrence probabilities.
* Mapped the mathematical clusters back to human-readable categories:
  * Topic 0: Business & Economy
  * Topic 1: Technology
  * Topic 2: Entertainment
  * Topic 3: Politics
  * Topic 4: Sports

## Key Insights & Edge-Case Testing
* **Data Prep is the Model:** Without aggressive lemmatization and noise filtering, LDA will simply cluster common verbs. In unsupervised learning, the quality of the preprocessing pipeline *is* the quality of the model.
* **The "Out-of-Domain" Trap:** To test the model's boundaries, I ran inference on a poetic, out-of-domain sentence: *"The peaceful ocean waves crashed gently against the golden sand under the moonlight."* * **Result:** The model categorized it as "Business & Economy." 
  * **Takeaway:** Because the model had only ever ingested BBC news data, it lacked the semantic space for "nature" or "poetry." This was a perfect hands-on demonstration that an AI's understanding of reality is strictly bounded by its training distribution.

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/AamirHussain12/news-categorizer-lda.git
   cd news-categorizer-lda
   ```

2. **Install the required dependencies**
   ```bash
   pip install pandas numpy scikit-learn nltk
   ```

3. **Run the Jupyter Notebook**
   
   Open Jupyter Notebook and run the notebook to see the step-by-step pipeline and topic extraction process.

   ```bash
   jupyter notebook
   ```

## Author

**Aamir Hussain**
