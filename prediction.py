# Import necessary libraries
from transformers import pipeline  # Hugging Face's interface
from tqdm import tqdm  # Library for displaying progress bars

# Define a function for performing inference using a pipeline object
def inference(pipe, prompt):
    result = pipe(prompt)  # Generate text based on the input prompt
    answer = result[0]['generated_text'].split("=")[-1]  # Extract the answer part after the "="
    return answer  # Return the extracted answer

# Define a function to predict sentiments based on a set of test data
def predict(X_test, model, tokenizer):
    y_pred = []  # List to store predicted labels
    # Initialize the pipeline for text generation with specified model and tokenizer
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=1,  # Generate only one new token per prompt
                    temperature=0.01,  # Very low temperature for deterministic outputs
                    # device='cuda'  # Uncomment to use GPU for acceleration
                    )
    # Iterate over each entry in the test dataset
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]  # Extract the text for the current test instance
        answer = inference(pipe, prompt)  # Perform inference to get the generated text

        # Determine the sentiment based on keywords in the generated text
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        elif "neutral" in answer:
            y_pred.append("neutral")
        else:
            y_pred.append("none")  # If no sentiment keyword is detected, label as 'none'
    return y_pred  # Return the list of predicted sentiments
