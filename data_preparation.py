# Import necessary libraries
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Define a function to generate training prompts including sentiment labels
def generate_prompt(row):
    return f"Review: {row['text']} [Sentiment: {row['sentiment']}]"

# Define a function to generate test prompts without sentiment labels
def generate_test_prompt(row):
    return f"Review: {row['text']}"

# Define the main function for preparing data
def prepare_data(filename="sentiment_analysis_part1.csv"):
    # Load data from a CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Lists to hold train and test data subsets
    X_train = list()
    X_test = list()
    # Stratify the data by sentiment and split into train and test sets
    for sentiment in ["positive", "neutral", "negative"]:
        train, test = train_test_split(df[df.sentiment == sentiment],
                                       train_size=100,
                                       test_size=100,
                                       random_state=20)
        X_train.append(train)
        X_test.append(test)

    # Concatenate the train and test subsets and shuffle the train subset
    X_train = pd.concat(X_train).sample(frac=1, random_state=10)
    X_test = pd.concat(X_test)

    # Create an evaluation dataset from unused instances
    eval_idx = [idx for idx in df.index if idx not in list(train.index) + list(test.index)]
    X_eval = df[df.index.isin(eval_idx)]
    # Randomly sample from each sentiment class to balance the evaluation set
    X_eval = (X_eval.groupby('sentiment', group_keys=False)
              .apply(lambda x: x.sample(n=50, random_state=10, replace=True)))
    # Reset index after shuffling
    X_train = X_train.reset_index(drop=True)

    # Apply the prompt generation function to training and evaluation data
    X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), columns=["text"])
    X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1), columns=["text"])

    # Extract true labels for the test set and apply the prompt generation function
    y_true = X_test.sentiment
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])

    # Convert training and evaluation DataFrames to Hugging Face Dataset objects
    train_data = Dataset.from_pandas(X_train)
    eval_data = Dataset.from_pandas(X_eval)

    # Return prepared data
    return train_data, eval_data, X_test, y_true
