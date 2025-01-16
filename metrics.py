import numpy as np
from sklearn.metrics import auc

# F2.1 - Expressive Power
def f2_1(n, F, C):
    """
    Calculate the Expressive Power (F2.1) score.

    Parameters:
        n (int): Number of distinct explanatory outputs.
        F (list): List of unique representation formats.
        C (set): Set of comprehensible formats defined for the end-user.

    Returns:
        float: Expressive Power score.
    """
    # Count how many formats in F are part of the comprehensible set C
    comprehensible_count = sum(1 for f in F if f in C)

    # Calculate the score using the formula
    score = n + len(F) + (comprehensible_count / len(F)) if F else 0  # Avoid division by zero if F is empty
    return round(score,1)

# F3- Selectivity
def f3(s, tunable=False, sigma=2):
    """
    Calculate the Selectivity (F3) score based on explanation size.

    Parameters:
        s (int): Explanation size (e.g., number of highlighted features).
        tunable_to_7 (bool): Whether the method allows tuning s (to 7 for example; it can tune to lower or higher).
        sigma (int, optional): Standard deviation for the Gaussian decay. Default is 2.

    Returns:
        float: Selectivity score (range: 0 to 1).
    """
    if tunable:
        return 1.0
    else:
        return round((2.718 ** (-((s - 7) ** 2) / (2 * sigma ** 2))), 1)
    

# F4.2 - Target Sensitivity
def f4_2(E1, E2, d_max, distance_metric):
    """
    Calculate Target Sensitivity (F4.2) score.
    !!! This is an example for tabular datasets, we have to create other functions for other data types !!!

    Parameters:
        E1 (array): Explanation before perturbation.
        E2 (array): Explanation after perturbation.
        d_max (float): Maximum possible distance for normalization.
        distance_metric (function): Function to compute the distance between E1 and E2.

    Returns:
        float: Normalized distance score (0 to 1).
    """
    d = distance_metric(E1, E2)
    return round(d / d_max,1)

# F6.2 - Surrogate Agreement
def f6_2(blackbox_preds, surrogate_preds):
    """
    Calculate the Surrogate Agreement (F6.2) score.

    Parameters:
        blackbox_preds (array): Black-box model predictions.
        surrogate_preds (array): Surrogate model predictions.

    Returns:
        float: Surrogate Agreement score (1 - average prediction difference) between 0 and 1.
    """
    N = len(blackbox_preds)  # Number of instances
    avg_diff = np.mean(np.abs(blackbox_preds - surrogate_preds))
    return round(1 - avg_diff,1)

# F7.1 - Incremental Deletion
def f7_1_get_probs_auc(model, instance, base_instance, feature_ranking, X):
    """
    Perform incremental deletion for a single instance and calculate AUC.
    
    Parameters:
        model: Trained black-box model.
        instance (array): Original instance (to be perturbed).
        base_instance (array): Optimal feature values for class 0.
        feature_ranking (list): List of features ranked by an XAI method.
        X (DataFrame): DataFrame containing feature names.
    
    Returns:
        tuple: (probabilities, auc_score)
            - probabilities: List of predicted probabilities at each step.
            - auc_score: Area under the curve for probability decay.
    """
    perturbed_instance = instance.copy()
    probabilities = [model.predict_proba(perturbed_instance.reshape(1, -1))[0][1]]  # Initial probability

    for feature in feature_ranking:
        # Replace the feature with its optimal value
        feature_index = X.columns.get_loc(feature)
        perturbed_instance[feature_index] = base_instance[0][feature_index]
        
        # Predict the probability for class 1
        prob = model.predict_proba(perturbed_instance.reshape(1, -1))[0][1]
        probabilities.append(prob)
    
    # Calculate AUC for probability decay
    auc_score = auc(range(len(probabilities)), probabilities)
    return probabilities, auc_score

def f7_1_result(auc_xai, auc_random):
    """
    Calculate the normalized F7.1 score for a single instance.
    
    Parameters:
        auc_xai (float): AUC of the XAI method.
        auc_random (float): AUC of the random explainer.
    
    Returns:
        float: Normalized F7.1 score (ranges from 0 to 1).
    """
    if auc_random == 0:
        raise ValueError("Random explainer AUC cannot be zero.")
    return round((auc_random - auc_xai) / auc_random,1)

# F7.2 - ROAR
def f7_2_classification(model, X_train, X_test, y_train, y_test, feature_ranking, random_ranking):
    """
    Calculate the normalized ROAR metric (F7.2) for classification tasks using feature shuffling.
    
    Parameters:
        model: Trained black-box model.
        X_train (DataFrame): Training dataset.
        X_test (DataFrame): Testing dataset.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.
        feature_ranking (list): Feature ranking provided by the XAI method.
        random_ranking (list): Random feature ranking.
    
    Returns:
        tuple: (Normalized F7.2 score, acc_xai, acc_random)
            - Normalized F7.2 score (float): Score ranging from 0 to 1.
            - acc_xai (list): List of accuracies after shuffling features ranked by the XAI method.
            - acc_random (list): List of accuracies after shuffling features ranked randomly.
    """
    n_features = len(feature_ranking)
    
    # Initialize accuracy lists with the initial accuracy of the model
    model_initial = model.fit(X_train, y_train)
    initial_accuracy = model_initial.score(X_test, y_test)
    acc_xai = [initial_accuracy]
    acc_random = [initial_accuracy]
    
    # Copy datasets to preserve original data
    X_train_shuffled_xai = X_train.copy()
    X_test_shuffled_xai = X_test.copy()
    X_train_shuffled_random = X_train.copy()
    X_test_shuffled_random = X_test.copy()
    
    for n in range(1, n_features + 1):
        # Shuffle top n features according to XAI ranking
        xai_features_to_shuffle = feature_ranking[:n]
        random_features_to_shuffle = random_ranking[:n]
        
        for feature in xai_features_to_shuffle:
            # Shuffle the feature values in both train and test sets
            X_train_shuffled_xai[feature] = np.random.permutation(X_train_shuffled_xai[feature].values)
            X_test_shuffled_xai[feature] = np.random.permutation(X_test_shuffled_xai[feature].values)
        
        for feature in random_features_to_shuffle:
            # Shuffle the feature values in both train and test sets
            X_train_shuffled_random[feature] = np.random.permutation(X_train_shuffled_random[feature].values)
            X_test_shuffled_random[feature] = np.random.permutation(X_test_shuffled_random[feature].values)
        
        # Train models and evaluate accuracy
        model_xai = model.fit(X_train_shuffled_xai, y_train)
        acc_xai.append(model_xai.score(X_test_shuffled_xai, y_test))
        
        model_random = model.fit(X_train_shuffled_random, y_train)
        acc_random.append(model_random.score(X_test_shuffled_random, y_test))
    
    # Calculate AUCs
    auc_xai = auc(range(len(acc_xai)), acc_xai)
    auc_random = auc(range(len(acc_random)), acc_random)
    
    # Normalize ROAR metric
    if auc_random == 0:
        raise ValueError("Random explainer AUC cannot be zero.")
    m_f7_2 = (auc_random - auc_xai) / auc_random
    
    return round(m_f7_2,1), acc_xai, acc_random

# F7.3 - White-box
def f7_3_compute_agreement(xai_explanation, true_coefficients):
    """
    Compute the mean agrrement (accuracy) of the XAI explanations compared to the ground-truth coefficients.

    Parameters:
        xai_explanation (list or array): Coefficients from the XAI method.
        true_coefficients (list or array): Ground-truth coefficients of the linear function.

    Returns:
        float: Mean accuracy as a percentage.
    """
    
    # Convert inputs to numpy arrays for easy computation
    xai_explanation = np.array(np.abs(xai_explanation))
    true_coefficients = np.array(np.abs(true_coefficients))
    
    # Handle zero coefficients - do exp
    valid_indices = (true_coefficients != 0) & (xai_explanation != 0)  # Ignore zero coefficients
    if not np.any(valid_indices):
        xai_explanation = np.array(np.exp(xai_explanation))
        true_coefficients = np.array(np.exp(true_coefficients))
    
    # Calculate the accuracy as the ratio of the smaller to the larger value
    accuracies = np.minimum(xai_explanation, true_coefficients) / np.maximum(xai_explanation, true_coefficients)  
    # Compute the mean accuracy
    mean_accuracy = round(np.mean(accuracies),2)
    
    return mean_accuracy

def f7_3_score(agreement):
    """
    Calculate the F7.3 score based on the agreement.

    Parameters:
        agreement (float): Agreement value between 0 and 1.

    Returns:
        int: F7.3 score (0 to 3).
    """
    if agreement >= 0.95:
        return 3  # Complete agreement
    elif 0.80 <= agreement < 0.95:
        return 2  # High agreement
    elif 0.60 <= agreement < 0.80:
        return 1  # Some agreement
    else:
        return 0  # No agreement