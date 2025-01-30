import numpy as np
from sklearn.metrics import auc
import pandas as pd
import matplotlib.pyplot as plt

##### F2.1 - Expressive Power ####
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

##### F3- Selectivity ####
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
        return round((np.e ** (-((s - 7) ** 2) / (2 * sigma ** 2))), 1)
    

##### F4.2 - Target Sensitivity ####
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

##### F6.2 - Surrogate Agreement ####
def f6_2(blackbox_preds, surrogate_preds):
    """
    Calculate the Surrogate Agreement (F6.2) score.
    To be added: max((bx)) for regression problems; in classification this is 1.

    Parameters:
        blackbox_preds (array): Black-box model predictions.
        surrogate_preds (array): Surrogate model predictions.

    Returns:
        float: Surrogate Agreement score (1 - average prediction difference) between 0 and 1.
    """
    N = len(blackbox_preds)  # Number of instances
    avg_diff = np.mean(np.abs(blackbox_preds - surrogate_preds))
    return round(1 - avg_diff,1)

##### F7 - Faithfulness ####

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
    
#### F9 - Stability ####

# Similarity - neighbors computing taken from shapash library

# From shapash
def _compute_distance(x1, x2, mean_vector, epsilon=0.0000001):
    """
    Compute distances between data points by using L1 on normalized data : sum(abs(x1-x2)/(mean_vector+epsilon))

    Parameters
    ----------
    x1 : array
        First vector
    x2 : array
        Second vector
    mean_vector : array
        Each value of this vector is the std.dev for each feature in dataset

    Returns
    -------
    diff : float
        Returns :math:`\\sum(\\frac{|x1-x2|}{mean\\_vector+epsilon})`
    """
    diff = np.sum(np.abs(x1 - x2) / (mean_vector + epsilon))
    return diff

# From shapash
def _compute_similarities(instance, dataset):
    """
    Compute pairwise distances between an instance and all other data points

    Parameters
    ----------
    instance : 1D array
        Reference data point
    dataset : 2D array
        Entire dataset used to identify neighbors

    Returns
    -------
    similarity_distance : array
        V[j] == distance between actual instance and instance j
    """
    mean_vector = np.array(dataset, dtype=np.float32).std(axis=0)
    similarity_distance = np.zeros(dataset.shape[0])

    for j in range(0, dataset.shape[0]):
        # Calculate distance between point and instance j
        dist = _compute_distance(instance, dataset[j], mean_vector)
        similarity_distance[j] = dist

    return similarity_distance

# From shapash
def _get_radius(dataset, n_neighbors, sample_size=500, percentile=95):
    """
    Calculate the maximum allowed distance between points to be considered as neighbors

    Parameters
    ----------
    dataset : DataFrame
        Pool to sample from and calculate a radius
    n_neighbors : int
        Maximum number of neighbors considered per instance
    sample_size : int, optional
        Number of data points to sample from dataset, by default 500
    percentile : int, optional
        Percentile used to calculate the distance threshold, by default 95

    Returns
    -------
    radius : float
        Distance threshold
    """
    # Select 500 points max to sample
    size = min([dataset.shape[0], sample_size])
    # Randomly sample points from dataset
    rng = np.random.default_rng(seed=79)
    sampled_instances = dataset[rng.integers(0, dataset.shape[0], size), :]
    # Define normalization vector
    mean_vector = np.array(dataset, dtype=np.float32).std(axis=0)
    # Initialize the similarity matrix
    similarity_distance = np.zeros((size, size))
    # Calculate pairwise distance between instances
    for i in range(size):
        for j in range(i, size):
            dist = _compute_distance(sampled_instances[i], sampled_instances[j], mean_vector)
            similarity_distance[i, j] = dist
            similarity_distance[j, i] = dist
    # Select top n_neighbors
    ordered_x = np.sort(similarity_distance)[:, 1 : n_neighbors + 1]
    # Select the value of the distance that captures XX% of all distances (percentile)
    return np.percentile(ordered_x.flatten(), percentile)

# From shapash
def find_neighbors(selection, dataset, model, mode, n_neighbors=10):
    """
    For each instance, select neighbors based on 3 criteria:

    1. First pick top N closest neighbors (L1 Norm + st. dev normalization)
    2. Filter neighbors whose model output is too different from instance (see condition below)
    3. Filter neighbors whose distance is too big compared to a certain threshold

    Parameters
    ----------
    selection : list
        Indices of rows to be displayed on the stability plot
    dataset : DataFrame
        Entire dataset used to identify neighbors
    model : model object
        ML model
    mode : str
        "classification" or "regression"
    n_neighbors : int, optional
        Top N neighbors initially allowed, by default 10

    Returns
    -------
    all_neighbors : list of 2D arrays
        Wrap all instances with corresponding neighbors in a list with length (#instances).
        Each array has shape (#neighbors, #features) where #neighbors includes the instance itself.
    """
    instances = dataset.loc[selection].values

    all_neighbors = np.empty((0, instances.shape[1] + 1), float)
    """Filter 1 : Pick top N closest neighbors"""
    for instance in instances:
        c = _compute_similarities(instance, dataset.values)
        # Pick indices of the closest neighbors (and include instance itself)
        neighbors_indices = np.argsort(c)[: n_neighbors + 1]
        # Return instance with its neighbors
        neighbors = dataset.values[neighbors_indices]
        # Add distance column
        neighbors = np.append(neighbors, c[neighbors_indices].reshape(n_neighbors + 1, 1), axis=1)
        all_neighbors = np.append(all_neighbors, neighbors, axis=0)

    # Calculate predictions for all instances and corresponding neighbors
    if mode == "regression":
        # For XGB it is necessary to add columns in df, otherwise columns mismatch
        predictions = model.predict(pd.DataFrame(all_neighbors[:, :-1], columns=dataset.columns))
    elif mode == "classification":
        predictions = model.predict_proba(pd.DataFrame(all_neighbors[:, :-1], columns=dataset.columns))[:, 1]

    # Add prediction column
    all_neighbors = np.append(all_neighbors, predictions.reshape(all_neighbors.shape[0], 1), axis=1)
    # Split back into original chunks (1 chunck = instance + neighbors)
    all_neighbors = np.split(all_neighbors, instances.shape[0])

    """Filter 2 : neighbors with similar blackbox output"""
    # Remove points if prediction is far away from instance prediction
    if mode == "regression":
        # Trick : use enumerate to allow the modifcation directly on the iterator
        for i, neighbors in enumerate(all_neighbors):
            all_neighbors[i] = neighbors[abs(neighbors[:, -1] - neighbors[0, -1]) < 0.1 * abs(neighbors[0, -1])]
    elif mode == "classification":
        for i, neighbors in enumerate(all_neighbors):
            all_neighbors[i] = neighbors[abs(neighbors[:, -1] - neighbors[0, -1]) < 0.1]

    """Filter 3 : neighbors below a distance threshold"""
    # Remove points if distance is bigger than radius
    radius = _get_radius(dataset.values, n_neighbors)

    for i, neighbors in enumerate(all_neighbors):
        # -2 indicates the distance column
        all_neighbors[i] = neighbors[neighbors[:, -2] < radius]
    return all_neighbors

# Extract neighbors for each instance
def prepare_neighbors(all_neighbors, feature_columns):
    """
    Prepare a list of 2D arrays containing feature values of neighbors for XAI methods.

    Parameters:
    ----------
    all_neighbors : list of 2D arrays
        Output from find_neighbors function.
    feature_columns : list
        List of feature column names (excludes distance and predictions).

    Returns:
    -------
    exp_input : list of 2D arrays
        Each array corresponds to the neighbors of a specific instance, with only feature columns.
    """
    exp_input = []
    for neighbors in all_neighbors:
        # Extract only feature columns (exclude distance and prediction columns)
        feature_data = neighbors[:, :len(feature_columns)]
        exp_input.append(feature_data)
    return exp_input


def f9_score(exp_neighbors, distance_metric, epsilon=1e-8, metric="similarity"):
    """
    Compute the m_f9.2 metric for similarity (higher value for lower distance) or identity (same instance instead of neighbors).

    Parameters:
    ----------
    exp_neighbors : list of 2D arrays
        Each array corresponds to the neighbors of a specific instance, with feature values.
    distance_metric : function
        Function to compute the distance between explanations.
    epsilon : float, optional
        Small value to avoid division by zero, by default 1e-8.
    metric : str, optional
        "similarity" or "identity", by default "similarity".

    Returns:
    -------
    m_f9.2 : float
        Average similarity or identity score.
    """
    total_similarity = 0
    instance_count = len(exp_neighbors)

    for neighbors in exp_neighbors:
        # For identity, compute mean and std of the original explanations (column 1) only once
        if metric == "identity":
            original_explanations = np.array([row[0] for row in neighbors])
            mean_col1 = np.mean(original_explanations, axis=0)
            std_col1 = np.std(original_explanations, axis=0)

        # Normalize the neighbors
        if metric == "similarity":
            normalized_neighbors = (neighbors - np.mean(neighbors, axis=0)) / (np.std(neighbors, axis=0) + epsilon)
        else:  # For identity
            normalized_neighbors = (neighbors - mean_col1) / (std_col1 + epsilon)

        # Use the first neighbor as the reference instance (x1)
        x1 = normalized_neighbors[0]
        r = len(normalized_neighbors) - 1  # Number of neighbors excluding the instance itself
        similarities = []

        for j in range(1, len(normalized_neighbors)):
            xj = normalized_neighbors[j]
            
            # Compute distance based on the selected distance_metric
            dist = distance_metric(x1, xj)  # e.g., Euclidean distance

            # Compute similarity as 1 / (1 + distance)
            similarity = 1 / (1 + dist)
            similarities.append(similarity)

        # Average similarity for this instance and its neighbors
        total_similarity += (1 / r) * np.sum(similarities)

    # Average across all instances
    m_f9_2 = total_similarity / instance_count
    return round(m_f9_2, 1)


def scatter_feature_values(exp_neighbors, instance_idx=None, metric="similarity"):
    """
    Create a scatter plot of feature values for a specific instance or all instances.

    Parameters:
    ----------
    exp_neighbors : list of 2D arrays
        Each array corresponds to the neighbors of a specific instance, with feature values.
    instance_idx : list, optional
        List of Indexes of the instances to plot. If None, plots all instances, by default None.
    metric : str
        "similarity" or "identity".
    """
    # Select the instance(s) to plot
    if instance_idx is not None:
        instances_to_plot = instance_idx
    else:
        instances_to_plot = range(len(exp_neighbors))

    for idx in instances_to_plot:
        neighbors = np.array(exp_neighbors[idx])  # Convert to NumPy array
        plt.figure(figsize=(12, 6))
        for feature_idx in range(neighbors.shape[1]):
            # Swap x and y: feature values on x-axis, neighbor/run indices on y-axis
            plt.scatter(neighbors[:, feature_idx], range(neighbors.shape[0]), label=f"Feature {feature_idx + 1}")

        plt.title(f"Scatter Plot Across Neighbors (Instance {idx})" if metric == "similarity"
                  else f"Scatter Plot Across Runs (Instance {idx})")
        plt.xlabel("ExplanationValues")
        plt.ylabel("Neighbors / Runs")
        plt.legend()
        plt.grid(True)

        # Ensure y-axis has only integer values
        plt.yticks(ticks=range(neighbors.shape[0]))

        plt.show()

#### F11 - Speed ####
def calculate_speed_score(runtime):
    """
    Calculate the speed score based on the explanation runtime.

    Parameters:
    runtime (float): Runtime of the explanation generation in seconds.

    Returns:
    int: Speed score (0-4).
    """
    if runtime > 10:
        return 0
    elif 5 < runtime <= 10:
        return 1
    elif 1 < runtime <= 5:
        return 2
    elif 0.1 < runtime <= 1:
        return 3
    elif runtime <= 0.1:
        return 4
