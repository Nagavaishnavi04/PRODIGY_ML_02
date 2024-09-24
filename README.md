# PRODIGY_ML_02

**K-Means Customer Segmentation**

**Project Overview**

This project applies K-Means Clustering to segment customers of a retail store based on their purchase history. Customer segmentation helps businesses identify distinct customer groups for targeted marketing strategies, personalized promotions, and better understanding of customer behavior. The dataset used contains information about customers' ages, annual income, and spending scores.

**Problem Statement:**

The task is to group customers into distinct clusters based on their purchasing behavior using the K-Means Clustering algorithm. We aim to classify customers based on features like age, annual income, and spending score, helping the business to focus on different customer segments more effectively.

**Libraries Used:**

The following libraries are used in the project:

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib**: For data visualization and plotting.
- **Seaborn**: For enhanced data visualizations.
- **Scikit-learn**: For applying machine learning algorithms like K-Means, and metrics like silhouette score.
- **PCA (Principal Component Analysis)**: Used for dimensionality reduction to visualize the clusters in 2D.

**Dataset**

The dataset used in this project can be found here on Kaggle.(https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) 

It contains 200 records with the following columns:

- **CustomerID**: Unique identifier for each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Annual Income (k$)**: Annual income of the customer in thousands of dollars.
- **Spending Score (1-100)**: A score assigned by the store based on the customer's purchasing behavior.

**Project Workflow**

**1. Data Preprocessing:**

Load the dataset using Pandas and inspect the data for missing values and inconsistencies.
Only the relevant numerical features (```Age```, ```Annual Income (k$)```, and ```Spending Score (1-100)```) are selected for clustering.
Features are standardized using **StandardScaler** from **Scikit-learn** to ensure they are on the same scale, which improves the performance of the K-Means algorithm.

**2. Finding Optimal Clusters (Elbow Method):**

We determine the optimal number of clusters using the **Elbow Method**. This method helps to select the best number of clusters by plotting the **Within-Cluster Sum of Squares (WCSS)** for a range of cluster counts.
The "elbow point" in the plot indicates the best number of clusters, which minimizes the WCSS without overfitting the data.

**3. Applying K-Means Clustering:**

After identifying the optimal number of clusters (in this case, 5), we apply the K-Means Clustering algorithm to segment the customers.
Each customer is assigned to a cluster based on the similarity of their attributes.

**4. Cluster Visualization (PCA):**

To visualize the clusters, we use PCA (Principal Component Analysis) to reduce the dimensionality of the data to two components.
A scatter plot is generated to visualize the customer segments in 2D, with distinct colors for each cluster.

**5. Evaluation:**

The quality of clustering is evaluated using the Silhouette Score. A score close to 1 indicates well-separated clusters, while a score closer to -1 indicates overlapping clusters.
For this dataset, the silhouette score is around 0.55, which shows reasonably good clustering.

**6. Saving the Results:**

The resulting clusters are saved to a new CSV file (Clustered_Customers.csv), with an additional column indicating the cluster assignment for each customer.

**Project Files**

- ```kmeans_customer_segmentation.py```: The main Python script containing all the steps for data preprocessing, K-means clustering, and visualization.

- ```Mall_Customers.csv```: The dataset used for customer segmentation (make sure this is either added or linked from an external source like Kaggle).

- ```Clustered_Customers.csv```: The output file containing customer data along with their cluster labels.

- ```README.md```: This file, which explains the project in detail.

**How to Run the Project**

**Prerequisites**

Make sure you have Python installed along with the required libraries. You can install the necessary libraries by running:

```pip install pandas numpy matplotlib seaborn scikit-learn```

**Steps to Run:**

1.Download the dataset from Kaggle or use the dataset provided in this repository (```Mall_Customers.csv```).

2.Run the script ```kmeans_customer_segmentation.py``` in your Python environment (e.g., Jupyter Notebook, VSCode, or terminal).

3.The script will perform the clustering and save the output as ```Clustered_Customers.csv```.

4.The clustering visualization will be displayed, and the silhouette score will be printed in the terminal.

**Project Output**

- Elbow Method Plot: Helps visualize the optimal number of clusters for K-means.
- Cluster Scatter Plot: A 2D scatter plot showing the clusters of customers after applying PCA.
- Clustered Customers File: A CSV file containing the original data along with an additional column indicating the cluster each customer belongs to.
- Silhouette Score: A score that evaluates the quality of the clustering.

**Conclusion**

This project demonstrates how K-Means Clustering can be used to group retail customers based on their spending habits. By understanding different customer segments, businesses can make informed decisions regarding targeted marketing, personalized services, and customer retention strategies.

**Future Improvements**

Experiment with different clustering algorithms like Hierarchical Clustering or DBSCAN to see if better segmentation can be achieved.
Incorporate more customer features like purchase frequency, product preferences, or regional data for enhanced segmentation.
