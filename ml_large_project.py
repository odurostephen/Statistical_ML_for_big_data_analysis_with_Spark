import os
import logging
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

# Suppress SciPy/NumPy warnings (Optional)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

# Initialize Spark session with optimized configurations
spark = SparkSession.builder \
    .appName("EnhancedCardioHealthAnalysis") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

# Set Spark log level to ERROR to reduce verbosity
spark.sparkContext.setLogLevel("ERROR")

# Suppress specific Spark/Hadoop warnings
logger = logging.getLogger("org.apache.spark")
logger.setLevel(logging.ERROR)

logger_native = logging.getLogger("org.apache.hadoop.util.NativeCodeLoader")
logger_native.setLevel(logging.ERROR)

def create_image_directory(base_dir='images'):
    """Create a directory to save images if it doesn't exist."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

def load_data(file_path):
    """Load CSV data into Spark DataFrame with correct delimiter."""
    df = spark.read.csv(file_path, header=True, inferSchema=True, sep=';')
    print(f"Data Loaded Successfully with {df.count()} records and {len(df.columns)} columns.")
    df.printSchema()  # Optional: Print schema for verification
    return df

def analyze_missing_values(df):
    """Analyze and report missing values in the DataFrame."""
    missing_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
    missing_pd = missing_counts.toPandas().T.rename(columns={0: 'Missing_Count'})
    missing_pd = missing_pd[missing_pd['Missing_Count'] > 0]
    if not missing_pd.empty:
        print("Missing Values Detected:")
        print(missing_pd)
    else:
        print("No Missing Values Detected.")
    return missing_pd

def impute_missing_values(df, missing_pd):
    """Impute missing values: Mean for numerical and Mode for categorical variables."""
    # Identify numerical and categorical columns
    num_cols = [field.name for field in df.schema.fields if field.dataType.typeName() in ['integer', 'double']]
    cat_cols = [field.name for field in df.schema.fields if field.dataType.typeName() == 'string']
    
    # Mean imputation for numerical columns
    for col_name in num_cols:
        mean_val = df.select(F.mean(col_name)).first()[0]
        df = df.fillna({col_name: mean_val})
        print(f"Imputed missing values in numerical column '{col_name}' with mean: {mean_val}")

    # Mode imputation for categorical columns
    for col_name in cat_cols:
        mode_val = df.groupBy(col_name).count().orderBy(F.desc("count")).first()
        if mode_val:
            mode = mode_val[0]
            df = df.fillna({col_name: mode})
            print(f"Imputed missing values in categorical column '{col_name}' with mode: {mode}")
    return df

def remove_duplicates(df):
    """Remove duplicate rows from the DataFrame."""
    initial_count = df.count()
    df = df.dropDuplicates()
    final_count = df.count()
    duplicates_removed = initial_count - final_count
    print(f"Duplicates Removed: {duplicates_removed}")
    return df

def visualize_data(df, num_cols, cat_cols, image_dir):
    """Perform exploratory data analysis with visualizations and save them as images."""
    # Convert to Pandas DataFrame for visualization (sample if data is large)
    sample_pdf = df.sample(fraction=0.1, seed=42).toPandas()
    
    # Interactive Box Plots for numerical features
    for col_name in num_cols:
        fig = px.box(sample_pdf, y=col_name, title=f'Box Plot of {col_name}')
        save_path = os.path.join(image_dir, f'box_plot_{col_name}.png')
        fig.write_image(save_path)
        print(f"Saved Box Plot for {col_name} at {save_path}")
    
    # Interactive Histograms for numerical features
    for col_name in num_cols:
        fig = px.histogram(sample_pdf, x=col_name, nbins=30, title=f'Histogram of {col_name}', marginal="box")
        save_path = os.path.join(image_dir, f'histogram_{col_name}.png')
        fig.write_image(save_path)
        print(f"Saved Histogram for {col_name} at {save_path}")
    
    # Interactive Bar Charts for categorical features
    for col_name in cat_cols:
        counts = sample_pdf[col_name].value_counts().reset_index()
        counts.columns = [col_name, 'Count']
        fig = px.bar(counts, x=col_name, y='Count', title=f'Bar Chart of {col_name}')
        save_path = os.path.join(image_dir, f'bar_chart_{col_name}.png')
        fig.write_image(save_path)
        print(f"Saved Bar Chart for {col_name} at {save_path}")
    
    # Correlation Matrix Heatmap using Seaborn
    numeric_pdf = sample_pdf[num_cols]
    plt.figure(figsize=(12, 10))
    correlation_matrix = numeric_pdf.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    save_path = os.path.join(image_dir, 'correlation_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Correlation Matrix Heatmap at {save_path}")

def encode_and_prepare_features(df, cat_cols):
    """Encode categorical variables and assemble features for modeling."""
    # StringIndexer for categorical variables
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid='keep') for col in cat_cols]
    
    # OneHotEncoder for categorical variables with dropLast=False to retain all categories
    encoders = [OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded", dropLast=False) for col in cat_cols]
    
    # Assemble feature columns
    encoded_cols = [f"{col}_encoded" for col in cat_cols]
    # Exclude 'id' and 'cardio' from numerical features
    num_cols = [field.name for field in df.schema.fields
                if field.dataType.typeName() in ['integer', 'double']
                and field.name not in ['cardio', 'id']]
    feature_cols = encoded_cols + num_cols
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")
    
    # Standard Scaling
    scaler = StandardScaler(inputCol="assembled_features", outputCol="features", withStd=True, withMean=False)
    
    # Define the pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
    pipeline_model = pipeline.fit(df)
    df_prepared = pipeline_model.transform(df)
    
    feature_names = []
    for col in cat_cols:
        distinct_values = df.select(col).distinct().count()
        # Convert the encoded vector to an array
        array_col = f"{col}_array"
        df_prepared = df_prepared.withColumn(array_col, vector_to_array(f"{col}_encoded"))
        # Extract each element of the array into a separate column
        for i in range(distinct_values):
            encoded_feature = f"{col}_encoded_{i}"
            df_prepared = df_prepared.withColumn(encoded_feature, F.col(array_col)[i])
            feature_names.append(encoded_feature)
        # Optionally, drop the intermediate array column to save space
        df_prepared = df_prepared.drop(array_col)
    
    # Add numerical feature names
    feature_names += num_cols
    
    print(f"Total number of features after encoding: {len(feature_names)}")
    
    return df_prepared, feature_names

def compute_correlation_matrix(df, feature_names, target, image_dir):
    """Compute and visualize the correlation matrix, then save it as an image."""
    assembler = VectorAssembler(inputCols=feature_names, outputCol="features_corr")
    vector_df = assembler.transform(df).select("features_corr")
    corr_matrix = Correlation.corr(vector_df, "features_corr").head()[0].toArray().tolist()
    corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_df, annot=False, fmt=".2f", cmap='coolwarm')
    plt.title('Full Correlation Matrix')
    save_path = os.path.join(image_dir, 'full_correlation_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Full Correlation Matrix Heatmap at {save_path}")
    
    # Identify top correlated features with the target variable
    if target not in corr_df.columns:
        print(f"Target variable '{target}' not found in the correlation matrix.")
        return []
    
    target_corr = corr_df[target].abs().sort_values(ascending=False)
    top_features = target_corr.index[1:6]  # Top 5 excluding target itself
    print(f"Top 5 Correlated Features with {target}:")
    print(target_corr.head(6))
    return top_features.tolist()

def plot_scatter_plots(df, features, target, image_dir):
    """Create scatter plots for top correlated features against the target and save them as images."""
    if not features:
        print("No top features to plot scatter plots.")
        return
    
    sample_pdf = df.sample(fraction=0.1, seed=42).toPandas()
    for feature in features:
        fig = px.scatter(sample_pdf, x=feature, y=target,
                         title=f'Scatter Plot of {feature} vs {target}',
                         trendline="ols")
        save_path = os.path.join(image_dir, f'scatter_{feature}_vs_{target}.png')
        fig.write_image(save_path)
        print(f"Saved Scatter Plot for {feature} vs {target} at {save_path}")

def feature_importance(df, feature_names, target, image_dir):
    """Train a Logistic Regression model and display feature importance by saving a bar plot."""
    # Split the data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Initialize Logistic Regression
    lr = LogisticRegression(featuresCol="features", labelCol=target, maxIter=10)
    
    # Pipeline
    pipeline = Pipeline(stages=[lr])
    model = pipeline.fit(train_df)
    
    # Get feature coefficients
    lr_model = model.stages[-1]
    coefficients = lr_model.coefficients.toArray()
    
    # Debugging: Print lengths
    print(f"Number of feature_names: {len(feature_names)}")
    print(f"Number of coefficients: {len(coefficients)}")
    
    # Ensure lengths match
    if len(feature_names) != len(coefficients):
        print("Error: The number of feature names does not match the number of coefficients.")
        print("Please verify the feature encoding process.")
        return
    
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['abs_coef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='abs_coef', ascending=False)
    
    # Plot Feature Importance
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df)
    plt.title('Feature Importance from Logistic Regression')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Features')
    save_path = os.path.join(image_dir, 'feature_importance_logistic_regression.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Feature Importance Plot at {save_path}")
    
    print("Feature Importance:")
    print(coef_df[['Feature', 'Coefficient']])

def evaluate_model(predictions, target):
    """Evaluate the model with various metrics."""
    evaluator_auc = BinaryClassificationEvaluator(labelCol=target, metricName="areaUnderROC")
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol=target, metricName="accuracy")
    evaluator_precision = MulticlassClassificationEvaluator(labelCol=target, metricName="weightedPrecision")
    evaluator_recall = MulticlassClassificationEvaluator(labelCol=target, metricName="weightedRecall")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=target, metricName="f1")
    
    auc = evaluator_auc.evaluate(predictions)
    accuracy = evaluator_accuracy.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    
    metrics = {
        "AUC": auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }
    
    return metrics

def plot_roc_curve(predictions, image_dir, model_name):
    """Plot ROC Curve and save as image."""
    roc = predictions.select("probability", "cardio").toPandas()
    roc['prob'] = roc['probability'].apply(lambda x: x[1])
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(roc['cardio'], roc['prob'])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    save_path = os.path.join(image_dir, f'roc_curve_{model_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC Curve for {model_name} at {save_path}")

def plot_confusion_matrix(predictions, target, image_dir, model_name):
    """Plot Confusion Matrix and save as image."""
    confusion_matrix = predictions.groupBy(target, 'prediction').count().toPandas()
    pivot_cm = confusion_matrix.pivot(index=target, columns='prediction', values='count').fillna(0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot_cm, annot=True, fmt='g', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    save_path = os.path.join(image_dir, f'confusion_matrix_{model_name}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Confusion Matrix for {model_name} at {save_path}")

def model_training(df, feature_names, target, image_dir):
    """Train multiple ML models with cross-validation and evaluate their performance."""
    # Split the data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Initialize Evaluators
    evaluator_auc = BinaryClassificationEvaluator(labelCol=target, metricName="areaUnderROC")
    
    # Dictionary to store models and their parameters
    models = {
        "LogisticRegression": {
            "estimator": LogisticRegression(featuresCol="features", labelCol=target, maxIter=20),
            "paramGrid": ParamGridBuilder() \
                .addGrid(LogisticRegression.regParam, [0.01, 0.1, 1.0]) \
                .addGrid(LogisticRegression.elasticNetParam, [0.0, 0.5, 1.0]) \
                .build()
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(featuresCol="features", labelCol=target, numTrees=100),
            "paramGrid": ParamGridBuilder() \
                .addGrid(RandomForestClassifier.numTrees, [50, 100, 200]) \
                .addGrid(RandomForestClassifier.maxDepth, [5, 10, 20]) \
                .build()
        },
        "SVM": {
            "estimator": LinearSVC(featuresCol="features", labelCol=target, maxIter=20),
            "paramGrid": ParamGridBuilder() \
                .addGrid(LinearSVC.regParam, [0.01, 0.1, 1.0]) \
                .build()
        },
        # DNN has been removed
    }
    
    results = {}
    
    for model_name, model_info in models.items():
        print(f"\nTraining and Evaluating {model_name}...")
        start_time = time.time()
        
        # Define Pipeline
        pipeline = Pipeline(stages=[model_info["estimator"]])
        
        # Define CrossValidator
        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=model_info["paramGrid"],
                                  evaluator=evaluator_auc,
                                  numFolds=5)
        
        # Train model with cross-validation
        cv_start_time = time.time()
        cv_model = crossval.fit(train_df)
        cv_end_time = time.time()
        cv_duration = cv_end_time - cv_start_time
        
        # Best model
        best_model = cv_model.bestModel
        print(f"Best Model Parameters for {model_name}:")
        for param, value in best_model.stages[-1].extractParamMap().items():
            print(f"  {param.name}: {value}")
        
        # Predictions
        pred_start_time = time.time()
        predictions = best_model.transform(test_df)
        pred_end_time = time.time()
        pred_duration = pred_end_time - pred_start_time
        
        # Evaluate Metrics
        metrics = evaluate_model(predictions, target)
        metrics['Training_Time_sec'] = cv_duration
        metrics['Prediction_Time_sec'] = pred_duration
        
        # Save Metrics
        results[model_name] = metrics
        print(f"Metrics for {model_name}: {metrics}")
        
        # Plot ROC Curve
        plot_roc_curve(predictions, image_dir, model_name)
        
        # Plot Confusion Matrix
        plot_confusion_matrix(predictions, target, image_dir, model_name)
    
    # Convert results to DataFrame for comparison
    results_df = pd.DataFrame(results).T
    results_df = results_df.reset_index().rename(columns={'index': 'Model'})
    print("\nModel Performance Comparison:")
    print(results_df)
    
    # Save results to CSV
    results_csv_path = os.path.join(image_dir, 'model_performance_comparison.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved Model Performance Comparison at {results_csv_path}")

def main():
    # Define the base directory for saving images
    image_dir = create_image_directory('images')
    
    # Corrected path to the dataset
    file_path = "cardio_train.csv"
    
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist. Please verify the path.")
        spark.stop()
        return
    
    # Load data
    df = load_data(file_path)
    
    # Analyze missing values
    missing_pd = analyze_missing_values(df)
    
    # Impute missing values if any
    if not missing_pd.empty:
        df = impute_missing_values(df, missing_pd)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Identify numerical and categorical columns
    num_cols = [field.name for field in df.schema.fields
                if field.dataType.typeName() in ['integer', 'double']
                and field.name not in ['cardio', 'id']]
    
    cat_cols = ['gender', 'cholesterol', 'gluc']
    
    # Exploratory Data Analysis
    visualize_data(df, num_cols, cat_cols, image_dir)
    
    # Encode categorical variables and prepare features
    df_prepared, feature_names = encode_and_prepare_features(df, cat_cols)
    
    # Define target variable
    target_variable = 'cardio'
    
    # Ensure the target variable exists
    if target_variable not in df_prepared.columns:
        print(f"Target variable '{target_variable}' not found in the dataset.")
        spark.stop()
        return
    
    # Compute and visualize correlation matrix
    top_features = compute_correlation_matrix(df_prepared, feature_names, target_variable, image_dir)
    
    # Scatter plots for top correlated features
    plot_scatter_plots(df_prepared, top_features, target_variable, image_dir)
    
    # Feature importance analysis using Logistic Regression
    feature_importance(df_prepared, feature_names, target_variable, image_dir)
    
    # Model Training and Evaluation
    model_training(df_prepared, feature_names, target_variable, image_dir)
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
