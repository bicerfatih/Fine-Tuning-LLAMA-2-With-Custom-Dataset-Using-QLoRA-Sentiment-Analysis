# main.py
import os
from init_model_MPS_AppleM2 import init_model
from data_preparation import prepare_data
from train import train
from dotenv import load_dotenv

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

load_dotenv('./.env')

def main():
    # Initialize the model and tokenizer
    model, tokenizer = init_model()

    # Prepare the data
    train_data, eval_data, X_test, y_true = prepare_data()

    # Perform training
    train(model, tokenizer, train_data, eval_data)

    print("Training completed successfully!")

if __name__ == "__main__":
    main()
