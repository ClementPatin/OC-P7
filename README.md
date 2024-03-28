# OC-P7
# TWEETS SENTIMENT ANALYSIS

## project scenario

Air Paradis, a fictitious airline, wishes to be supported in the development of an AI product to anticipate bad buzz on social networks.

## proposed solution

In order to **respond as best as possible to *Air Paradis*** and allow **better evangelization of the knowledge acquired**, we have broken down the mission as follows:
- Tests of several **modeling** approaches:
    - **"Simple tailor-made model"**: rapid development of a **classic** machine learning model (logistic regression for example), which will serve as a basis for comparison
    - **"Advanced custom model"**: further development of a Deep Learning model (bidirectional LSTM).<br>
      Tests of:
        - several preprocessing methods
        - several **word embeddings**
    - the contribution of a model based on transformers: ***BERT*** (*Bidirectional Encoder Representation from Transformers*)
- Development of the prototype following the **MLOps approach**:
    - summary presentation of the **principles of MLOps** and its contributions
    - use of **`MLFlow`** for:
        - ensure **experiment management** (tracking, reporting)
        - centralize **model storage**
        - test the proposed **serving**
    - implement a **continuous deployment pipeline** of the winning model:
        - via an **API**: Git + GitHub + Azure WebApp
        - integrating **unit tests**
        - **limiting costs**:
            - **free** version of Azure WebApp
            - limit the size of the winning model via **TensorFlow Lite**
- Write a **blog article** on the modeling approach used:
    - models
    - metrics
    - interpretability
    - MLOps approach
- Prepare **a presentation** explaining the approach with **popularization work**


## organization of the github repository

```bash

│   .gitignore
│   Patin_Clement_2_scripts_012024.ipynb
│   README.md
│   logo_air_paradis.png
│   myFunctions.py
│   requirements.txt
├───.devcontainer
│   ...
├───.github
│   └───workflows
│   ...
├───mlruns
│   ...
├───mySaves
│   ├───advanced_model_large
│   │       final_advanced.keras
│   │       final_advancedcustom_standardize_args.joblib
│   │
│   ├───lists
│   │       contractions.joblib
│   │       emails.joblib
│   │       emoticons.joblib
│   │       escapes.joblib
│   │       hashtags.joblib
│   │       mentions.joblib
│   │       repeatedCharsWords.joblib
│   │       urls.joblib
│   │
│   ├───mlflow_ui_plots
│   │       parallel_coord_29_worst.png
│   │       parallel_coord_baseline.png
│   │
│   ├───plots
│   │       contractionsWordCloud.jpg
│   │       emailsWordCloud.jpg
│   │       emojisWordCloud.jpg
│   │       emoticonsWordCloud.jpg
│   │       escapesWordCloud.jpg
│   │       hashtagsWordCloud.jpg
│   │       mentionsWordCloud.jpg
│   │       number_of_tokens_plot.png
│   │       repeatedCharsWordCloud.jpg
│   │       urlsWordCloud.jpg
│   │       wordcloudNeg.jpg
│   │       wordcloudPos.jpg
│   │       wordsFreqsByTarget.png
│   │
│   ├───shap
│   │       force_plot_2.png
│   │       force_plot_42.png
│   │
│   └───TfLite
│           custom_standardize_args.joblib
│           ltsm_model_TFLite.tflite
│           test_roc_pr_curves.png
│           text_vectorizer.keras
│
├───Patin_Clement_1_modele_012024
│   ├───backend
│   │   │   .dockerignore
│   │   │   Dockerfile
│   │   │   main.py
│   │   │   myFunctionsForApp.py
│   │   │   requirements.txt
│   │   │   test_main.py
│   │   ├───TfLite
│   │   │       custom_standardize_args.joblib
│   │   │       ltsm_model_TFLite.tflite
│   │   │       test_roc_pr_curves.png
│   │   │       text_vectorizer.keras
│   └───frontend
│           Dockerfile
│           requirements.txt
│           streamlit_app.py
│
├───TestDockerCompose
│   │   docker-compose.yml
│   ├───backend
│   │   │   .dockerignore
│   │   │   Dockerfile
│   │   │   main.py
│   │   │   myFunctionsForApp.py
│   │   │   requirements.txt
│   │   │   test_main.py
│   │   ├───TfLite
│   │   │       custom_standardize_args.joblib
│   │   │       ltsm_model_TFLite.tflite
│   │   │       test_roc_pr_curves.png
│   │   │       text_vectorizer.keras
│   └───frontend
│           Dockerfile
│           requirements.txt
│           streamlit_app.py
```

## other needed files and folders in local

`Patin_Clement_2_scripts_012024.ipynb` needs other files and folders to run :

```bash

├───dataset
│       training.1600000.processed.noemoticon.csv
├───pretrained_embeddings
│       fasttext-wiki-news-300d-1M.vec
│       glove.6B.300d.txt
│       glove.twitter.27B.50d.txt
│
```