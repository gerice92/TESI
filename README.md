# Platform for complex to plain language transformation

This repository contains the source code for the final project for TESI. The following dependencies are required for it to run as expected, assuming that **pip** is the preferred package manager for Python:

```bash
pip install numpy scipy scikit-learn scrapy nltk
```

## Architecture

This platform is divided into relatively independent modules, which are called in sequence to provide the language transformation functionality.

### Step 0: Train the complex word classifier (Only once)

> Source: classifier/word\_classifier.py

Evaluate and train classifiers for different feature extraction setups, generating a file that can be imported from Python as a ready-to-use classifier. This way, the classifier is trained only once and it can be used from other programs as long as the input conforms to its expectations. This trained complex word classifier will be used in **step 2**.

### Step 1: Crawl web articles

> Source: crawler/

Extract plain text from HTML articles on the web, as well as other elements that help keeping the site's experience (e.g. images, styles). Plain text (split into sentences) goes directly to **step 2** to be processed, other elements go to **step 4** as there is no need to transform or process them.

### Step 2: Classify words as complex or simple

> Source: classifier/word\_classifier.py

Use the trained complex word classifier with a sentence as input and return a list of complex words on the sentence.

### Step 3: Find simple synonyms for complex words

> Source: synonyms/synonym\_replace.py

From a sentence, a list of complex words and their respective start and end positions in the sentence, map each complex word with a simple synonym and then replace each occurrence in the sentence.

### Step 4: Generate and open accesible web page

> Source: web/web\_generator.py, web/web\_launcher.py

Create a web page using text output from **step 5** and other elements from **step 1** that keep the site's experience. Open a web browser tab with this page, using (if possible) the default program for this task.