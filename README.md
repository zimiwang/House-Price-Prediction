# House Price Prediction Using Regression Techniques

## For more information please read the report

## Project Overview

This project encompasses a comprehensive analysis of house price prediction using regression techniques and an exploration into deep learning applications within NLP. It aims to provide insights into the real estate market through predictive modeling and to advance the understanding of textual data processing through deep learning models.

## Motivation

- **House Price Prediction:** The project seeks to address the complexity of the real estate market by accurately predicting house prices, thereby aiding stakeholders in making informed decisions.
- **NLP with Deep Learning:** The aim is to leverage deep learning models to process and summarize large volumes of textual data efficiently, demonstrating the potential of NLP in extracting meaningful information from text.

## Project Workflow

### House Price Prediction

1. **Data Collection and Preparation:** Utilizing the Ames Housing dataset, the project begins with data importation, cleaning to handle missing values and outliers, and preprocessing to convert categorical variables to numerical formats for analysis.
   
2. **Feature Selection and Analysis:** A correlation-based feature selection method is employed to identify key attributes significantly impacting house prices, such as the overall quality and living area of the house.
   
3. **Model Implementation:** Several regression models, including Linear Regression, L1/L2 Regularization, and Gradient Boosting, are applied. The models are evaluated based on their Root Mean Square Error (RMSE), with Gradient Boosting Regression identified as the most effective model due to its ability to handle nonlinear relationships and missing data.

### Natural Language Processing

1. **Dataset Integration and Cleaning:** The project leverages the "CNN/DailyMail" dataset, focusing on cleaning and preparing the text data for processing, including text normalization and tokenization.
   
2. **Deep Learning Model Development:** Recurrent Neural Networks (RNNs) are implemented to summarize news articles. The project explores various architectures and optimization techniques to improve model performance.
   
3. **Evaluation and Analysis:** The models' effectiveness is assessed based on their ability to generate coherent and concise summaries, highlighting the challenges and solutions in applying deep learning to NLP tasks.

## Technical Details

- **Technologies Used:** Python, Pandas, NumPy, TensorFlow, Scikit-learn, Matplotlib, Seaborn
- **Key Concepts:** Regression Analysis, Clustering Algorithms, RNNs, Text Summarization, Feature Engineering
- **Development Tools:** Jupyter Notebooks for an iterative and explorative approach to data analysis and model development

## Getting Started

Clone the repository to access the datasets and Jupyter Notebooks containing the project's code. Ensure the installation of all required libraries as listed in the `requirements.txt` file. Follow the notebooks for step-by-step instructions on data preprocessing, model training, and evaluation.

## Future Work

The project sets the stage for further exploration into more advanced models and techniques, such as Convolutional Neural Networks (CNNs) for image-based predictions and Transformer models for more sophisticated NLP tasks.

## License

This project is made available under the MIT License.

## Project Team

**Ziming Wang**
- Role: Coordination of research efforts and primary development of the model Implementation (Clustering).
- Affiliation: Graduate Student, University of Utah

**Cheng Zhen**
- Role: Coordination of research efforts and primary development of the model Implementation (Regression).
- Affiliation: Undergraduate Student, University of Utah
