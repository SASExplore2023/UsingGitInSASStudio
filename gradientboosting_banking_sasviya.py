# %% [markdown]
# ## GradientBoostingClassifier from SAS® Viya® on Banking
#
# ### Source
# This example is adapted from [Random Forest Classification with Scikit-Learn](https://www.datacamp.com/tutorial/random-forests-classifier-python) by Adam Shafi

# %% [markdown]
# ### About the Data Set:
#
# The data set is sourced from the UCI Machine Learning repository and pertains to direct marketing campaigns (phone calls) conducted by a Portuguese banking institution. The objective of classification is to predict whether a client will subscribe (1/0) to a term deposit (variable y).
#
# The data set contains customer information, comprising 41,188 records and 21 fields.

# %%
import os
import pandas as pd
import numpy as np

# %%
print(os.getcwd())
workspace = os.getcwd() + '/UsingGitInSASStudio/data/'
df = pd.read_csv(workspace + "banking_raw.csv")

df.head()

# %% [markdown]
# ### Gradient Boosting Workflow
#
# To fit and train this model, we will do the following:
#
# * Split the data
# * Train the model
# * Hyperparameter tuning
# * Assess model performance
#
# Note that there is no need to to convert all non-numeric features (e.g., month, education) into numeric ones for model fitting. The GradientBoostingClassifier provides native support for categorical features.

# %% [markdown]
# ### Split into predictor and response dataframes

# %%
X_df = df.drop('y', axis=1)
y = df['y']

X_df.shape,y.shape

# %%
y.value_counts()

# %% [markdown]
# ### Splitting the Data
#
# When training any supervised learning model, it is important to split the data into training and test data. The training data is used to fit the model. The algorithm uses the training data to learn the relationship between the features and the target. The test data is used to evaluate the performance of the model.
#
# The code below splits the data into separate variables for the features and target, then splits into training and test data.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size = 0.2, random_state = 10)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# ### Fitting and Evaluating the Model
#
# We first create an instance of the gradient boosting model, with the default parameters. We then fit this to our training data. We pass both the features and the target variable, so the model can learn.
#
# For details about using the `GradientBoostingClassifier` class in the `sasviya` package, see the [GradientBoostingClassifier documentation](https://documentation.sas.com/?cdcId=workbenchcdc&cdcVersion=default&docsetId=explore&docsetTarget=n1kiea90s0276wn1xr0ig0hvkix6.htm).

# %%
from sasviya.ml.tree import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100,
                                max_depth=5,
                                min_samples_leaf=1,
                                max_features=None,
                                learning_rate = 0.1,
                                subsample = 1.0,
                                random_state=0,
                                calc_feature_importances=True)
gb.fit(X_train, y_train)

# %% [markdown]
# At this point, we have a trained gradient boosting model, but we need to find out whether it is making accurate predictions.

# %%
y_pred = gb.predict(X_test)

# %% [markdown]
# The simplest way to evaluate this model is using accuracy; we check the predictions against the actual values in the test set and count up how many the model got right.

# %%
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred.astype(type(y_test[0])))
print(f"Accuracy: {accuracy:.2f}")

# %% [markdown]
# ### ROC Curve

# %%
from sklearn import metrics
import matplotlib.pyplot as plt

probs = gb.predict_proba(X_test).to_numpy()
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='Gradient Boosting (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### Confusion Matrix
# Let’s look at the confusion matrix. This plots what the model predicted against what the correct prediction was. We can use this to understand the tradeoff between false positives (top right) and false negatives(bottom left) We can plot the confusion matrix using this code:

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate test set predictions with the model
y_pred = gb.predict(X_test).astype(type(y_test[0]))

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();

# %% [markdown]
# ### Plot Feature Importances

# %%
feature_importances = gb.feature_importances_.set_index('Variable')['Importance']

# Set the figure size
plt.figure(figsize=(10, 6))  # Adjust the width (10) and height (6) as desired

# Plot a simple bar chart
feature_importances.plot.bar();

# Add labels and title
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance');
