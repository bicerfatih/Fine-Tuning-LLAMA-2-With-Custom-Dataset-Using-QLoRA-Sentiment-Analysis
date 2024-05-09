# Fine-Tuning-LLAMA-2-With-Custom-Dataset-Using-QLoRA-Sentiment-Analysis
Fine Tuning LLAMA 2 on Custom Dataset Using QLoRA and Sentiment Analysis

The project includes scripts and services for Fine-tuning LLMs on Custom Dataset such as data preparation, initializing the model, training a sentiment analysis model and making predictions. 

Fine-tuning basically means training a model on a specific task like Q&A or use cases. 

I’m aiming to fine-tune a model to analyse financial data from company reports, specifically using QLoRa for sentiment classification.

## Project Structure

- `create_sentiment_data.py` - Script to create or preprocess sentiment data.
- `data_preparation.py` - Script for preparing the data for training and testing.
- `init_model_MPS_AppleM2.py` - Initializes the sentiment analysis model using Apple's Metal Performance Shaders (MPS) on an M2 chip.
- `main.py` - The main script that ties all components of the project together.
- `prediction.py` - Script used to make predictions using the trained model.
- `sentiment_analysis_part1.csv` - Sample dataset used for sentiment analysis.
- `sentiment_analysis_part2.csv` - Sample dataset used for sentiment analysis.
- `train.py` - Script for training the sentiment analysis model.

## Installation

To run the scripts, clone the repository and install the required packages:

```bash
git clone [repository-url]
cd [repository-directory]
pip install -r requirements.txt
```

## Usage

To run the main program, execute:

```bash
python main.py
```

Ensure you have adjusted the paths and configurations as necessary for your specific setup.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

