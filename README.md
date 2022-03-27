# nlp_case_study

## 1) preprocessing
0) install dependencies from requirements.txt
1) (optional) register at deepl.com to get your personal auth key and save it in a project folder in deepl_authkey.txt file
2) (optional) run preprocessing.py
   1) translate into english
   2) tokenization
   3) lemmatization / stemming
   4) feature extraction
   > How to treat multiple text fields?  
   > _FeatureUnion_ vs _ColumnTransformer_
   > > 1st step: ColumnTransformer to get numeric data of all fields
   > > 2nd step: eventually use FeatureUnion for e.g., combining PCA and SelectKBest
   > 1) merge all in one string?
   > 2) using a token for each columns (not sure how)
   > 3) Using different dense layers (or embedding) for each column and concatenate them.
   
   3) vectorization
   4) 1-4 can be done with _TfidfVectorizer_
      1) alternatives: Count Vectorizer / Word2Vec
   5) dimensionality reduction: apply e.g. [sklearn.decomposition.TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) 
   > The NaiveBayes classifier needs discrete-valued features, but the PCA breaks this property of the features. You will have to use a different classifier if you want to use PCA.


## 2) training model
   1) train model
      1) [sklearn.linear_model.SGDClassifier](sklearn.linear_model.SGDClassifier) (stochastic gradient descent)
      2) SVM
      3) naive bayes
      4) random decision tree 
   2) Parameter tuning with the help of GridSearchCV
   3) Try other classification Algorithms Like 
      1) Linear Classifier, 
      2) Boosting Models and 
      3) even Neural Networks.
    
   **_Linear classifiers (SVM, logistic regression, etc.) with SGD training._**
> This estimator implements regularized linear models with stochastic gradient descent (SGD) learning: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate). SGD allows minibatch (online/out-of-core) learning via the partial_fit method. For best results using the default learning rate schedule, the data should have zero mean and unit variance.

