# nlp_case_study
## task:
1. Create two kinds of classifiers for this task and compare their results. You may
use any model architecture, statistical method, NLP algorithm or other method to
create these models,
   > 4 classifiers were tested (using scikit lib):
   1) [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) This estimator implements regularized linear models with stochastic gradient descent (SGD) learning,
   2) [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html?highlight=multinomialnb#sklearn.naive_bayes.MultinomialNB) Naive Bayes classifier for multinomial models is  suitable for classification with discrete features,
   3) [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) Method as a subset of support vector machine: C-Support Vector Classification.
   4) [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforest#sklearn.ensemble.RandomForestClassifier) A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
2. Compare the results of the models using whatever metric you believe is relevant
for this type of task, and
3. Make the code and models available for our team to review in either a python
script, Jupyter notebook, Pycharm project, or other accessible platform. The code
should be well-documented with step-by-step explanation of your approach to the
classification task. Basic visualizations can be included with titles and legends,
but color and style don’t matter.

## info
the code with preprocessing and model training and fine-tunning is available in [processing.py](https://github.com/buddhaha/nlp_case_study/blob/main/processing.py). Summary is shown in [show_results.ipynb](https://github.com/buddhaha/nlp_case_study/blob/main/show_results.ipynb)

## 1) prepare
1) clone repo
> git clone https://github.com/buddhaha/nlp_case_study.git
2) install dependencies from requirements.txt
3) save data 'Relevant vs Irrelevant.xlsx' in raw_data/ folder
## 2) preprocessing
2) (optional) register at deepl.com to get your personal auth key and save it in a project folder in deepl_authkey.txt file
3) (optional) use translate_w_deepl from preprocessing.py to translate the text
   1) translate into english
   2) tokenization
   3) lemmatization / stemming
   4) feature extraction on combined text fields (feature aggregation with ColumnTransformer)
      1) _TfidfVectorizer_ was used as it showed pretty good results (_CountVectorizer_ or others shoud be tested)
      2) dimensionality reduction might be needed e.g. [sklearn.decomposition.TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) 
   > NOTE: The NaiveBayes classifier needs discrete-valued features, but the PCA breaks this property of the features. You will have to use a different classifier if you want to use PCA.


## 3) training the model
   1) train model
   > see [processing.py](https://github.com/buddhaha/nlp_case_study/blob/main/processing.py) for detailed setting of each model 
   2) Parameter fine-tuning with the help of GridSearchCV
## 4) evaluation
   All classifiers performed pretty well. Classification report for each classifier used can be seen in [show_results.ipynb](https://github.com/buddhaha/nlp_case_study/blob/main/show_results.ipynb).
   As we want to minimize *false positive* (FP), from definition of *precision* (precision= TP/(TP+FP)) we seek to approach 1 or 100% with precision.

![alt text](https://github.com/buddhaha/nlp_case_study/blob/main/perf_comparison.png?raw=True)


## 5) improvments @TODO:
   1) translate text with some api and implement following preprocessing pipeline: tokenization, stopwords removal, lemmatization; se we dont have to throw half of the data away.
   2) multilang classifiers / feature extractor?
   
   
