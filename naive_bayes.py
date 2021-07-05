"""
Class and methods for a Naive Bayes Classifier (Gaussian) model. Assumes the class-conditional marginal probability 
densities (feature_i | class_i) of each feature are Gaussian (normal) distributions.
"""

# Import libraries, packages, modules needed:
import math
import numpy as np
import statistics
import sys


# Class object for our Naive Bayes Classifier (Gaussian):
class NaiveBayesGaussian:
    """
    A Naive Bayes Classifier model. Assumes the class-conditional marginal probability densities 
    (feature_i | class_i) of each feature are Gaussian (normal) distributions.
    """
    def __init__(self):  # priors= option to input prior probabilities as in sklearn version, kde= , smoothing= )
        self.classes = None
        self.num_features = None
        self.class_priors = None
        self.ccps_likelihoods = None

    def get_class_priors(self, y_target):
        """
        Calculate and save class priors (the standalone marginal probability of each class).
        """
        self.classes = list(set(y_target))
        self.class_priors = {Class: np.count_nonzero(y_target == Class) / len(y_target) for Class in self.classes}
        return self.class_priors
    
    def get_ccps_likelihoods(self, X_features, y_target):
        """
        Calculate the summary statistics (mean, sample variance) for the class-conditional probability (feature_i | class_i) 
        distributions of each feature, from the training data provided when fitting the model.
        """
        self.num_features = X_features.shape[1]
        # Create an empty dict to add the summary stats to:
        self.ccps_likelihoods = {}
        # Add the summary stats for each (feature_i | class_i) class-conditional probability distribution to 
        # our ccps_likelihoods dict:
        for Class in self.classes:
            current_class_only = np.array([X_features[row_num, :] for row_num in range(len(y_target)) if y_target[row_num] == Class])
            # Add current class to dict as key, with value as an empty array of the right size (# of features):
            self.ccps_likelihoods[Class] = [None for feature_num in range(self.num_features)]
            # Add the summary stats for this class, for each feature:
            for feature_num in range(self.num_features):
                current_feature_intersection_class = current_class_only[:, feature_num]
                self.ccps_likelihoods[Class][feature_num] = {"class": Class, 
                                                             "feature_number": feature_num, 
                                                             "mean": np.mean(current_feature_intersection_class), 
                                                             "variance": np.var(current_feature_intersection_class, ddof=1)}
        # [?? To do: Check to make sure no unfilled/empty dicts {} in the dict ??]
        return self.ccps_likelihoods

    def get_conditional_prob_gaussian(self, for_class, feature_num, value):
        """
        Calculates the class-conditional probability p(feature_i = value | class_i) for the 
        given feature, class and value. 
        
        Assumptions: 
        Uses the summary statistics for the class-conditional probability distribution of 
        (feature_i | class_i) from the training data used to fit() the model, and assumes 
        it is a Gaussian (normal) distribution.
        """
        # Get mean and sample variance for the class-conditional probability (feature_i | class_i) 
        # distributions of this feature, from the training data provided when fitting the model.
        mean = self.ccps_likelihoods[for_class][feature_num]["mean"]
        variance_sample = self.ccps_likelihoods[for_class][feature_num]["variance"]
        # Calculate p(feature_i = value | class_i) using a Gaussian (normal) probability distribution:
        e_exponent = -(value - mean)**2 / (2 * variance_sample)
        conditional_prob = (1 / (math.sqrt(2 * math.pi * variance_sample))) * math.exp(e_exponent)

        return conditional_prob
    
    def get_bayes_numerator(self, input, for_class):
        """
        Calculate the numerator of Bayes' Theorem for all features for the given class. This numerator is what our 
        Naive Bayes model (predict() method below) compares to determine which class to predict for a given input.
        """
        product_of_conditionals = 1
        for feature_num in range(self.num_features):
            # Get Gaussian conditional probability: p(feature = value | class):
            p_feature_given_class = self.get_conditional_prob_gaussian(for_class=for_class, 
                                                                       feature_num=feature_num, 
                                                                       value=input[feature_num]
                                                                       )
            # Get product of the above conditionals for every feature for this class: Bayes' Theorem, rewritten:
            # p(x1=val1, x2=val2, ..., xn=val_n | class_n) = p(x1 | class_n) * p(x2 | class_n) * ... * p(xn | class_n)
            product_of_conditionals *= p_feature_given_class

        # Multiply the product of the conditional by p(class) to get the numerator of Bayes' Theorem:
        p_class = self.class_priors[for_class]
        bayes_numerator = product_of_conditionals * p_class

        return bayes_numerator
    
    def fit(self, X_features, y_target):
        # Convert input X_train feature matrix and y_train target vector (these can be lists, numpy 
        # arrays, pandas dataframes/series) into numpy arrays to standardize the input types:
        X_features = np.array(X_features)
        y_target = np.array(y_target)
        # Handle case where y_target is a Pandas dataframe with only 1 column:
        if y_target.ndim == 2:
            y_target = y_target.reshape((len(y_target),))
        # Make sure y_target is 1-dimensional array:
        try:
            assert y_target.ndim == 1
        except AssertionError as error:
            sys.exit("Error: Please provide a y_target with 1 dimension.")

        # Calculate and save class priors (the standalone marginal probability of each class):
        self.get_class_priors(y_target)

        # Calculate the summary statistics (mean, sample variance) for the class-conditional probability 
        # (feature_i | class_i) distributions of each feature, from the training data provided:
        self.get_ccps_likelihoods(X_features, y_target)
    
    def predict(self, X_features):
        # Convert input X_train feature matrix and y_train target vector (these can be lists, numpy 
        # arrays, pandas dataframes/series) into numpy arrays to standardize the input types:
        X_features = np.array(X_features)
        
        # Check to make sure the inputs are the right size (same # of features as the training data):
        try:
            assert X_features.shape[1] == self.num_features
        except AssertionError:
            print(f"""Error: Please make sure your X_features data for the predict() method has the 
            same number of features as the training data used to fit the model.""")
            sys.exit()
        
        # Create array of right size (but full of NaNs) to add our predictions to:
        predictions = [None for input_num in range(X_features.shape[0])]

        # Get prediction for each input data point (row), and add to our predictions array:
        for input_num in range(X_features.shape[0]):
            class_probs = {}
            for Class in self.classes:
                p = self.get_bayes_numerator(input=X_features[input_num], for_class=Class)
                class_probs[Class] = p
            class_with_highest_prob = max(class_probs, key=class_probs.get)
            predictions[input_num] = class_with_highest_prob
            
            # # [?? To do: REMOVE these print statements ??]
            # print(f"Row/input #: {input_num}")
            # print(f"class_probs: \n{class_probs}")
            # print(f"class_with_highest_prob: {class_with_highest_prob} (prob: {class_probs[class_with_highest_prob]})\n")
        
        # Return the predictions array:
        return predictions

    def score(self, X_features, y_true):        
        # Convert provided y_true (this can be a list, numpy array, pandas dataframes/series) 
        # into a numpy array to standardize the types:
        y_true = np.array(y_true)
        # Handle case where y_target is a Pandas dataframe with only 1 column:
        if y_true.ndim == 2:
            y_true = y_true.reshape((len(y_true),))

        # Get predictions for input data (X_features a.k.a. X_test):
        predictions = self.predict(X_features=X_features)
        
        # Make sure y_target is a 1-dimensional array with the right length:
        try:
            assert y_true.ndim == 1
            assert len(predictions) == len(y_true)
        except AssertionError as error:
            sys.exit("""Error: Please make sure the provided y_true has 1 dimension, and has the 
            same length (number of observations or rows) as X_features.""")
        
        # Get accuracy score:
        accuracy_score = sum([predictions[item_num] == y_true[item_num] for item_num in range(len(predictions))]) / len(predictions)

        return accuracy_score
