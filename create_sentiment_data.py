import PyPDF2
import spacy
from textblob import TextBlob
import csv
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the English tokenizer from spaCy
nlp = spacy.load("en_core_web_sm")

# Function to find all PDF files
def find_pdf_files(root_folder):
    """Finds all PDF files within the root_folder and its subdirectories."""
    return glob.glob(f'{root_folder}/**/*.pdf', recursive=True)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join([page.extract_text() or '' for page in reader.pages])
        return text
    except Exception as e:
        logging.error(f"Error reading {pdf_path}: {e}")
        return None

# Function to segment sentences
def segment_sentences(text):
    """Segments text into sentences using spaCy."""
    try:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    except Exception as e:
        logging.error("Error in segmenting sentences: " + str(e))
        return []

# Function to analyze sentiment
def analyze_sentiment(sentences):
    """Analyzes sentiment of each sentence using TextBlob."""
    sentiments = []
    for sentence in sentences:
        sentiment = TextBlob(sentence).sentiment.polarity
        sentiments.append((sentence, 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral'))
    return sentiments

# Function to save results to CSV
def save_to_csv(sentiments, filename):
    """Saves the list of sentences with sentiments to a CSV file."""
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as file:  # Append to the file
            writer = csv.writer(file)
            writer.writerow(["text", "sentiment"])
            for sentence, sentiment in sentiments:
                writer.writerow([sentence, sentiment])
    except Exception as e:
        logging.error("Error saving to CSV: " + str(e))

# Main function to process PDFs and alternate between two files
def process_pdfs(pdf_files):
    output_csv_1 = "sentiment_analysis_part1.csv"
    output_csv_2 = "sentiment_analysis_part2.csv"
    half_index = len(pdf_files) // 2

    for index, pdf_path in enumerate(pdf_files):
        logging.info(f"Processing {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        if text:
            sentences = segment_sentences(text)
            sentiments = analyze_sentiment(sentences)
            if index < half_index:
                save_to_csv(sentiments, output_csv_1)
                logging.info("Data saved to " + output_csv_1)
            else:
                save_to_csv(sentiments, output_csv_2)
                logging.info("Data saved to " + output_csv_2)

if __name__ == "__main__":
    root_folder = "./pdf_chat_data"  # Directory containing PDFs
    pdf_files = find_pdf_files(root_folder)
    process_pdfs(pdf_files)
