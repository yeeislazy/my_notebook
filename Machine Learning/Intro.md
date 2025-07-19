# **Basic Workflow**

1. **Import Libraries**:

   - Import Scikit-learn and other necessary libraries like NumPy and Pandas.
2. **Load Data**:

   - Use datasets from Scikit-learn or load your own data using Pandas.
3. **Preprocess Data**:

   - Handle missing values, scale features, and encode categorical variables.
4. **Split Data**:

   - Split the dataset into training and testing sets using `train_test_split`.
5. **Train a Model**:

   - Choose a machine learning algorithm and train it on the training data.
6. **Evaluate the Model**:

   - Use metrics like accuracy, precision, recall, or mean squared error to evaluate the model.

---

# Some concept in machine learning

## 1. **Overfitting**:

- machine learning model learns not only the underlying pattern in training data but also the noise and random fluctuations, results in **high accuracy on training data** but  **poor performance on unseen (test) data** .
- *Signs of Overfitting*

  - **Low training error but high test error** : The model performs exceptionally well on training data but poorly on new data.
  - **Large gap between training and validation accuracy** : If training accuracy is high (e.g., 98%) but validation accuracy is much lower (e.g., 70%), overfitting is likely.
  - **Too complex a model for the given data** : The model captures unnecessary details, like noise, rather than general patterns.
- **Causes of Overfitting**

  | Cause                                  | Explanation                                                                       |
  | -------------------------------------- | --------------------------------------------------------------------------------- |
  | **Too many features**            | The model captures noise rather than general patterns.                            |
  | **Too complex a model**          | Deep trees, high-degree polynomials, or excessive parameters increase complexity. |
  | **Too few training examples**    | The model doesn’t generalize well due to lack of diverse data.                   |
  | **Too many training iterations** | The model memorizes rather than learns general trends.                            |
- **Prevent Overfitting**

  | Method                                  | Description                                                                         |
  | --------------------------------------- | ----------------------------------------------------------------------------------- |
  | **More Training Data**            | A larger dataset helps the model generalize better.                                 |
  | **Feature Selection**             | Remove irrelevant or redundant features to simplify the model.                      |
  | **Regularization (L1/L2)**        | Adds penalties (Ridge/Lasso) to prevent excessive complexity.                       |
  | **Cross-Validation**              | Splitting data into multiple parts (e.g., K-Fold) ensures robustness.               |
  | **Early Stopping**                | Stops training when validation error starts increasing.                             |
  | **Dropout (for Neural Networks)** | Randomly disables neurons during training to prevent reliance on specific features. |
  | **Ensemble Methods**              | Combining multiple models (e.g., Bagging, Boosting) improves generalization.        |

## 2. **Robustness**:

- refers to a model's ability to maintain accuracy and performance even when faced with ***Noisy or outlier data***, ***Unseen or slightly different data distributions***, ***Adversarial attacks or small changes in inputs*** and ***Missing values or corrupted data***, ensures that the model generalizes well and is resistant to overfitting or bias caused by data variations.
- **Improve Robustness**| **Technique**    | **How It Helps**                                                          |
  | ---------------------- | ------------------------------------------------------------------------------- |
  | Regularization (L1/L2) | Prevents overfitting, ensures model stability                                   |
  | Cross-Validation       | Tests model on different subsets to check generalizability                      |
  | Data Augmentation      | Introduces variability in training data to make the model adaptable             |
  | Outlier Detection      | Removes extreme values that may skew predictions                                |
  | Robust Scalers         | Uses methods like Median Absolute Deviation (MAD) instead of mean-based scaling |
  | Adversarial Training   | Trains the model on slightly perturbed data to resist attacks                   |

## 3. **Regularization**:

- a technique used in machine learning to **prevent overfitting** by adding a **penalty term** to the loss function. It helps models generalize better by discouraging overly complex patterns that fit noise instead of actual data trends.
- Regularization **adds a penalty** to large coefficients to avoid complex models that memorize training data instead of learning true patterns.
- benifits:

  - **Prevents Overfitting**: Reduces the model’s dependence on training data.
  - **Improves Generalization**: Makes the model perform well on unseen data.
  - **Reduces Variance**: Ensures predictions remain stable even with noisy inputs.
- **Types of Regularization**

  | **Regularization Type**       | **Mathematical Form**                                | **Effect**                                                |
  | ----------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------- |
  | **L1 Regularization (Lasso)** | \( Loss + \lambda \sum \|w_i\| \)                          | Shrinks some weights to**exact zero** (feature selection) |
  | **L2 Regularization (Ridge)** | \( Loss + \lambda \sum w_i^2 \)                            | Shrinks weights**closer to zero** but not exactly zero    |
  | **Elastic Net (L1 + L2)**     | \( Loss + \lambda_1 \sum \|w_i\| + \lambda_2 \sum w_i^2 \) | Combines L1 & L2; useful when features are correlated           |
  | **Dropout (Neural Networks)** | Randomly**drops neurons** during training            | Reduces co-dependency, improves generalization                  |

  **λ (lambda)** controls the strength of regularization:


  - **If λ is small**, regularization is weak (risk of overfitting).
  - **If λ is too large**, regularization is too strong (risk of underfitting).
- **When to Use Regularization?**

  - Your model performs **very well on training data** but **poorly on test data** (overfitting).
  - You have **high-dimensional data** (many features).
  - You want to **avoid complex models** and ensure feature selection.
- **Example**

```python
   from sklearn.linear_model import Ridge, Lasso
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_squared_error
   import numpy as np

   # Sample Data
   X = np.random.rand(100, 5)
   y = X[:, 0] * 5 + X[:, 1] * 3 + np.random.randn(100)  # y depends on X[:,0] and X[:,1]

   # Train-Test Split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Ridge Regression (L2)
   ridge = Ridge(alpha=0.1)
   ridge.fit(X_train, y_train)
   y_pred_ridge = ridge.predict(X_test)
   print("Ridge MSE:", mean_squared_error(y_test, y_pred_ridge))

   # Lasso Regression (L1)
   lasso = Lasso(alpha=0.1)
   lasso.fit(X_train, y_train)
   y_pred_lasso = lasso.predict(X_test)
   print("Lasso MSE:", mean_squared_error(y_test, y_pred_lasso))
```

- **Choosing Between L1 and L2 Regularization**
  - **Use L1 (Lasso) if:**✔ You want **feature selection** (some coefficients become exactly **zero**).✔ You suspect **only a few features** are important.
  - **Use L2 (Ridge) if:**✔ You want **all features** to contribute but with **smaller weights**.✔ Your data has **high multicollinearity** (strongly correlated features).
  - **Use Elastic Net if:**
    ✔ Your dataset has **many correlated features** and you want both feature selection and weight shrinkage.

## 4. **Multicollinearity**:

- occurs when two or more independent variables in a regression model are highly correlated, meaning they provide redundant or overlapping information about the dependent variable.
- This can make it difficult to determine the individual effect of each independent variable on the dependent variable, as their contributions become intertwined.
- ***Key Issues with Multicollinearity***
  1. **Unstable Coefficients**: The regression coefficients may become highly sensitive to small changes in the data.
  2. **Reduced Interpretability**: It becomes challenging to interpret the effect of each independent variable.
  3. **Inflated Variance**: Standard errors of the coefficients increase, leading to less reliable statistical tests.
- ***Detecting Multicollinearity***
  - **Variance Inflation Factor (VIF)**: A VIF value greater than 10 is often considered a sign of high multicollinearity.
  - **Correlation Matrix**: High pairwise correlations between independent variables can indicate multicollinearity.
- ***Addressing Multicollinearity***
  - Use **Ridge Regression** or **Lasso Regression**, which apply regularization to reduce the impact of multicollinearity.
  - Remove or combine highly correlated variables.
  - Perform dimensionality reduction techniques like **Principal Component Analysis (PCA)**.

## 5. **Decomposition**:

- refers to techniques used to break down a dataset or a matrix into simpler components. These techniques are often used for **dimensionality reduction**, **feature extraction**, or **data compression**.
- Decomposition methods are particularly useful when working with high-dimensional data, as they help reduce the number of features while retaining the most important information.
- ***Common Decomposition Techniques***

  | **Technique**                               | **Description**                                                                                                |
  | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
  | **Principal Component Analysis (PCA)**      | Reduces dimensionality by finding the directions (principal components) that maximize variance.                      |
  | **Singular Value Decomposition (SVD)**      | Factorizes a matrix into three matrices:\( U \), \( \Sigma \), and \( V^T \). Used in PCA and other applications.    |
  | **Non-Negative Matrix Factorization (NMF)** | Decomposes a matrix into non-negative components. Useful for feature extraction.                                     |
  | **Latent Dirichlet Allocation (LDA)**       | A probabilistic decomposition technique used for topic modeling in text data.                                        |
  | **Independent Component Analysis (ICA)**    | Decomposes data into statistically independent components. Useful for signal separation and blind source separation. |


  1. **Principal Component Analysis (PCA)**
     - **Purpose**: Dimensionality reduction by projecting data onto a lower-dimensional space.
     - **How it Works**:

       - Finds the directions (principal components) that maximize variance in the data.
       - Projects the data onto these components.
     - **Use Case**: Reducing the number of features while retaining most of the variance.

       **Example: PCA**

       ```python
       from sklearn.decomposition import PCA
       import numpy as np

       # Sample Data
       X = np.random.rand(100, 5)

       # Apply PCA
       pca = PCA(n_components=2)  # Reduce to 2 dimensions
       X_reduced = pca.fit_transform(X)

       print("Explained Variance Ratio:", pca.explained_variance_ratio_)
       ```
  2. **Singular Value Decomposition (SVD)**
     - **Purpose**: Factorizes a matrix \( A \) into three matrices: \( U \), \( \Sigma \), and \( V^T \).
       \[
       A = U \Sigma V^T
       \]
     - **How it Works**:
     - \( U \): Left singular vectors.
     - \( \Sigma \): Singular values (diagonal matrix).
     - \( V^T \): Right singular vectors.
     - **Use Case**: Used in PCA, dimensionality reduction, and recommendation systems.
       **Example: SVD**
       ```python
       from sklearn.decomposition import TruncatedSVD
       import numpy as np

       # Sample Data
       X = np.random.rand(100, 5)

       # Apply SVD
       svd = TruncatedSVD(n_components=2)  # Reduce to 2 dimensions
       X_reduced = svd.fit_transform(X)

       print("Explained Variance Ratio:", svd.explained_variance_ratio_)
       ```
  3. **Non-Negative Matrix Factorization (NMF)**
     - **Purpose**: Decomposes a matrix into non-negative components.
     - **How it Works**:
     - Factorizes a matrix \( A \) into two matrices \( W \) and \( H \), where all elements are non-negative:
       \[
       A \approx W \times H
       \]
     - **Use Case**: Feature extraction, image processing, and text mining.
       **Example: NMF**
       ```python
       from sklearn.decomposition import NMF
       import numpy as np

       # Sample Data
       X = np.abs(np.random.rand(100, 5))  # Ensure non-negative data

       # Apply NMF
       nmf = NMF(n_components=2, random_state=42)
       W = nmf.fit_transform(X)
       H = nmf.components_

       print("W Shape:", W.shape)
       print("H Shape:", H.shape)
       ```
  4. **Latent Dirichlet Allocation (LDA)**
     - **Purpose**: A probabilistic decomposition technique used for topic modeling in text data.
     - **How it Works**:
     - Decomposes a document-term matrix into topics and their associated word distributions.
     - **Use Case**: Topic modeling in natural language processing (NLP).
       **Example: LDA**

       ```python
       from sklearn.decomposition import LatentDirichletAllocation
       import numpy as np

       # Sample Data (Document-Term Matrix)
       X = np.random.randint(0, 10, (100, 10))  # 100 documents, 10 terms

       # Apply LDA
       lda = LatentDirichletAllocation(n_components=2, random_state=42)
       lda.fit(X)

       print("Topic-Word Distribution:", lda.components_)
       ```
  5. **Independent Component Analysis (ICA)**
     - **Purpose**: Decomposes data into statistically independent components.
     - **How it Works**:
     - Finds components that are statistically independent from each other.
     - **Use Case**: Signal processing, such as separating audio signals (e.g., blind source separation).
       **Example: ICA**
       ```python
       from sklearn.decomposition import FastICA
       import numpy as np

       # Sample Data
       X = np.random.rand(100, 5)

       # Apply ICA
       ica = FastICA(n_components=2, random_state=42)
       X_reduced = ica.fit_transform(X)

       print("Independent Components Shape:", X_reduced.shape)
       ```
- **Key Points**

  - **Decomposition**   is used for dimensionality reduction, feature extraction, and data compression.
  - Common techniques include PCA, SVD, NMF, LDA, and ICA.
  - Each technique has specific use cases:
  - **PCA**: Dimensionality reduction while retaining variance.
  - **SVD**: Matrix factorization for recommendation systems and PCA.
  - **NMF**: Non-negative data decomposition for feature extraction.
  - **LDA**: Topic modeling in text data.
  - **ICA**: Signal processing and blind source separation.

## 6. **Pareto theory**

- The **Pareto Principle**, also known as the **80/20 Rule**, states that roughly 80% of effects come from 20% of causes. In machine learning, this can be applied to feature selection, model performance, and data quality.
- **Applications in Machine Learning**:
  - **Feature Selection**: Identify the 20% of features that contribute to 80% of the model's performance.
  - **Model Performance**: Focus on the 20% of hyperparameters that significantly impact the model's accuracy.
  - **Data Quality**: Address the 20% of data issues that cause 80% of errors in predictions.
  - **Error Analysis**: Identify the 20% of errors that account for 80% of the model's misclassifications.

## 7. **Evaluation**:

- **Evaluation** is the process of assessing the performance of a machine learning model using various metrics and techniques. It helps determine how well the model generalizes to unseen data and identifies areas for improvement.
- **Key Evaluation Metrics**:| **Metric**                   | **Description**                                                                                  |
  | ---------------------------------- | ------------------------------------------------------------------------------------------------------ |
  | **Accuracy**                 | Proportion of correct predictions to total predictions.                                                |
  | **Precision**                | Proportion of true positive predictions to total positive predictions.                                 |
  | **Recall (Sensitivity)**     | Proportion of true positive predictions to actual positives.                                           |
  | **F1 Score**                 | Harmonic mean of precision and recall, balancing both metrics.                                         |
  | **ROC-AUC**                  | Area under the Receiver Operating Characteristic curve, measuring model performance across thresholds. |
  | **Mean Squared Error (MSE)** | Average squared difference between predicted and actual values (for regression).                       |
  | **R-squared (R²)**          | Proportion of variance explained by the model (for regression).                                        |

## 8. **Cross-Validation**:

- A technique to assess model performance by splitting data into multiple subsets (folds) and training/testing on different combinations.

## 9. **Data discretization**:

- The process of converting continuous data into discrete categories or bins. This can help simplify models and improve interpretability.
- **Methods**:
  - **Equal Width Binning**: Divides the range of data into equal-width intervals.
  - **Equal Frequency Binning**: Divides data into bins with an equal number of observations.
  - **Clustering-based Binning**: Uses clustering algorithms to group similar data points.

## 10. Feature Scaling:

- is a technique to standardize the independent features present in the data, to scale highly vary values into same scale
- Why important:
  - different scales can bias algorithms:
    - distance-based algorithms will be dominated by features with larger magnitudes
    - linear regression algorithms are sensitive to feature scales as coefficients are inversely related to feature scale
  - gradient descent optimization: features with different scales can cause
    - **Slow convergence** : The algorithm takes longer to find the optimal solution
    - **Oscillating behavior** : The cost function may bounce around
    - **Numerical instability** : Very large or small values can cause computational issues
- When to scale:
  - Distance-based algorithms:
    - **K-Means Clustering** : Uses Euclidean distance
    - **K-Nearest Neighbors (KNN)** : Distance-based predictions
    - **Support Vector Machines (SVM)** : Especially with RBF kernel
    - **Neural Networks** : Gradient descent optimization
  - Linear regression and its variants:
    - **Linear Regression** : Coefficients are sensitive to feature scales
    - **Ridge Regression** : Regularization is affected by feature scales
    - **Lasso Regression** : Feature selection is influenced by scales
    - **Elastic Net** : Combines L1 and L2 regularization, sensitive to scales
- When not to scale:
  - **Tree-based algorithms** : Random Forest, Decision Trees, XGBoost
  - **Naive Bayes** : Assumes feature independence
  - **Some regression models** : When feature magnitudes have meaningful interpretations
- Common scaling techniques:
  - StandardScaler (Z-score normalization): Mean = 0, Std = 1
    - Best for normally distributed data and algorithms assuming Gaussian distribution, can handle outliers better than MinMaxScaler.
    - use when:
      - Data is normally distributed
      - No significant outliers
      - using algorithms that assume Gaussian distribution (SVM, Neural Networks)
      - Need interpretable results (mean=0, std=1)
    - avoid when:
      - Data has significant outliers
      - Using algorithms sensitive to outliers (KNN, K-Means)
      - Data is not normally distributed
      - `from sklearn.preprocessing import StandardScaler`
  - MinMaxScaler: Scales to range [0, 1]
    - Best for bounded features, neural networks, but not robust to outliers.
    - use when:
      - Need bounded range [0,1]
      - Neural networks (bounded activation functions)
      - Data has known min/max bounds
      - Preserving exact relative distances is important (eg, distance-based clustering, image data & computer vision,geospatial data,time series data)
    - avoid when:
      - Data has significant outliers
      - Future data might exceed current min/max
    - `from sklearn.preprocessing import MinMaxScaler`
  - RobustScaler: Uses median and IQR (less sensitive to outliers)
    - Best for datasets with outliers, preserves relative relationships.
    - use when:
      - Data contains outliers
      - Distribution is skewed
      - Features have different scales
      - Need robust scaling (eg, financial data, sensor data)
    - avoid when:
      - Data is perfectly normal without outliers
      - Need specific range bounds
    - `from sklearn.preprocessing import RobustScaler`
  - MaxAbsScaler: Scales by maximum absolute value
    - Best for sparse data, preserves sparsity, already centered data.
    - use when:
      - Data is sparse (many zeros)
      - Data is already centered around zero
      - Working with text data (TF-IDF)
      - Need to preserve sparsity (eg, text data, image data)
    - avoid when:
      - Data has significant outliers
      - Need specific range bounds
    - `from sklearn.preprocessing import MaxAbsScaler`
  - Summary:
    - Start with StandardScaler for most cases
    - Use RobustScaler if you detect outliers
    - Use MinMaxScaler for neural networks or when you need bounded ranges
    - Use MaxAbsScaler for sparse data or when preserving zeros is important

## 11. Discretization

- converts continuous features into discrete categories or bins, simplifying models and improving interpretability.
  - Why important:
    - Simplifies complex relationships, reduces complexity of continuous data.
    - Noise reduction, reduces the impact of outliers and noise.
    - Non-linear relationships, captures non-linear patterns by grouping similar values.
    - Improves interpretability, makes models easier to understand and visualize.
    - Allows categorical modeling, enables the use of algorithms that require categorical inputs （eg. Naive Bayes, certain decision trees）
  - Methods:
    - **Equal Width Binning**: Divides the range of values into equal-width intervals.
      - use when:
        - Data is uniformly distributed
        - You want to maintain equal intervals
        - Interpretability is important
      - avoid when:
        - Data has outliers or skewed distribution
        - You need adaptive binning
    - **Equal Frequency Binning**: Divides data into bins with an equal number of observations.
      - use when:
        - Data is skewed or has outliers
        - You want balanced representation in each bin
        - Adaptive binning is needed
      - avoid when:
        - Data is uniformly distributed
        - You need fixed-width intervals
    - **Clustering-based Binning**: Uses clustering algorithms (e.g., K-Means) to group similar values.
      - use when:
        - Data has complex patterns or clusters
        - You want adaptive binning based on data distribution
        - Non-linear relationships exist
      - avoid when:
        - Data is simple or uniformly distributed
        - You need fixed-width or frequency-based bins

        ``` python
        from sklearn.preprocessing import KBinsDiscretizer
        import numpy as np

        # Create sample data
        np.random.seed(42)
        sample_data = np.random.normal(50, 15, 1000).reshape(-1, 1)

        # Different strategies
        strategies = ['uniform', 'quantile', 'kmeans']
        encodings = ['onehot', 'ordinal', 'onehot-dense']

        for strategy in strategies:
            print(f"\nStrategy: {strategy}")
            print("-" * 15)
            
            discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy=strategy)
            data_discretized = discretizer.fit_transform(sample_data)
            
            # Show bin edges
            bin_edges = discretizer.bin_edges_[0]
            print(f"Bin edges: {bin_edges.round(2)}")
            
            # Show distribution
            unique, counts = np.unique(data_discretized, return_counts=True)
            print(f"Bin counts: {counts}")
            
            # Show some examples
            print("Examples:")
            for i in range(5):
                original = sample_data[i, 0]
                discretized = data_discretized[i, 0]
                print(f"  {original:.2f} → Bin {int(discretized)}")
        ```

    - **Decision Tree-based Binning**: Uses decision trees to create bins based on feature importance.
      - use when:
        - Data has complex interactions between features
        - You want adaptive, data-driven binning
        - Non-linear relationships exist
      - avoid when:
        - Data is simple or uniformly distributed

## 12. Regression model

- Linear regression: assuming the relationship between features and target as a straight line (or hyperplane in multiple dimensions).

  - use when:
    - Target variable is continuous
    - Relationship between features and target is approximately linear
    - You need interpretable results
    - Simple baseline model
    - Small to medium datasets
  - avoid when:
    - Non-linear relationships
    - High multicollinearity among features
    - Outliers significantly affect results
    - Need feature selection or regularization
- Logistic regression: predicts the probability of a binary (or categorical) outcome using the logistic function (sigmoid function) .

  - use when:
    - Binary classification (spam/not spam, fraud/not fraud)
    - Multi-class classification (with extensions)
    - Need probability estimates
    - Linear decision boundary is appropriate
    - Interpretable results needed
- Ridge regression: linear regression with L2 regularization to prevent overfitting by penalizing large coefficients.

  - use when:
    - High-dimensional data (many features)
    - Multicollinearity among features
    - Overfitting in linear regression
    - Want to keep all features but reduce their impact
    - Stable, robust predictions needed
  - avoid when:
    - Need feature selection (use Lasso instead)
    - Very few features
    - Perfect linear relationship with no noise
- Lasso regression: linear regression with L1 regularization that can shrink coefficients to exactly zero, performing automatic feature selection. Less stable than Ridge with correlated features.

  - use when:
    - High-dimensional data (many features),many irrelevant features suspected
    - Feature selection is important
    - Want sparse models (fewer features)
    - Multicollinearity among features (less stable than Ridge)
    - Want to reduce model complexity
  - avoid when:
    - All features are important
    - Non-linear relationships exist
    - Need to keep all features (use Ridge instead)
- ElasticNet regression: combines L1 and L2 regularization, balancing feature selection and multicollinearity handling.

  - use when:
    - High-dimensional data with many correlated features
    - Need both feature selection and multicollinearity handling
    - Want a balance between Ridge and Lasso
    - Some features are irrelevant, but others are important
  - avoid when:
    - All features are important (use Ridge)
    - Non-linear relationships exist (use other models)
    - Need simple linear regression without regularization
- Comparison Summary：

| **Model**               | **Target Type** | **Regularization** | **Feature Selection** | **Multicollinearity** | **Interpretability** |
| ----------------------------- | --------------------- | ------------------------ | --------------------------- | --------------------------- | -------------------------- |
| **Linear Regression**   | Continuous            | None                     | No                          | Poor                        | High                       |
| **Logistic Regression** | Binary/Categorical    | None                     | No                          | Moderate                    | High                       |
| **Ridge**               | Continuous            | L2                       | No                          | Excellent                   | High                       |
| **Lasso**               | Continuous            | L1                       | Yes                         | Moderate                    | High                       |
| **ElasticNet**          | Continuous            | L1 + L2                  | Yes                         | Good                        | High                       |

- Decision Flow Chart：

```
Start
  ↓
Is target continuous?
  ↓ Yes                    ↓ No
Linear Regression    →  Logistic Regression
  ↓
Overfitting?
  ↓ Yes                    ↓ No
Need feature selection?    →  Use Linear Regression
  ↓ Yes          ↓ No
  Lasso       →  Ridge
  ↓
Grouped features?
  ↓ Yes          ↓ No
ElasticNet    →  Lasso
```

dummy()
label encoder()
      one hot encoder()
