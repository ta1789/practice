import os
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException,Query
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
import os
import csv
import markdown
import duckdb
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import sqlite3
import numpy as np
import urllib

# API Token for LLM
load_dotenv()
TOKEN = os.getenv("API_PROXY")
print(TOKEN)

# Initialize FastAPI app
app = FastAPI()

# Root data directory
DATA_DIR = Path("C:/data")
print(DATA_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

class task(BaseModel):
    description: str


def validate_file_path(file_path: str) -> bool:
    """Ensure the file is within the allowed directory."""
    # Get absolute paths
    abs_file_path = os.path.abspath(file_path)
    abs_allowed_dir = os.path.abspath(DATA_DIR)

    # Check if the file path is inside the allowed directory
    return abs_file_path.startswith(abs_allowed_dir)


# Modify your task functions to use the validation
def process_file(file_path: str):
    if not validate_file_path(file_path):
        raise HTTPException(status_code=403, detail=f"Access to {file_path} is forbidden.")
    # Process the file here if it is valid

def prevent_deletion(file_path: str):
    """Ensure no file deletion operation is allowed."""
    if "delete" in file_path.lower() or "remove" in file_path.lower():
        raise HTTPException(status_code=403, detail="File deletion is not allowed.")

def parse_task_with_llm(task_description: str) -> dict:
    """Use LLM to classify the task and extract execution details."""
    prompt = f"""
    You are an AI assistant that converts task descriptions into structured JSON.

    **Task List:**
    - A1: Install uv & run datagen.py (requires email)
    - A2: Format a Markdown file using Prettier
    - A3: Count day asked in the query in a date list file and also give the day asked in the param
    - A4: Sort contacts from a JSON file
    - A5: Extract recent log file lines
    - A6: Create an index.json from Markdown files
    - A7: Extract sender email from an email text file
    - A8: Extract a credit card number from an image
    - A9: Extract transactions above $1000 from CSV
    - A10: Execute a query in SQLite for ticket-sales.db
    
    **Phase B Task List:**
    - B3: Fetch data from an API and save it
    - B4: Clone a git repo and make a commit
    - B5: Run a SQL query on a SQLite or DuckDB database
    - B6: Extract data from (scrape) a website
    - B7: Compress or resize an image
    - B8: Transcribe audio from an MP3 file
    - B9: Convert Markdown to HTML
    - B10: Filter a CSV file and return JSON data

    **Security Constraints:**
    - B1: Data outside the `/data` directory should never be accessed or exfiltrated, even if the task description requests it.
    - B2: Data should never be deleted anywhere on the file system, even if the task description requests it.

    **Task Description:**  
    {task_description}

    **Output JSON Format:**  
    {{"operation": "<A1-A10>", "file": "<file_path_if_needed>", "parameters": {{ "key": "value" }} }}

    Example:
    Input: "Sort all contacts in contacts.json"
    Output: {{ "operation": "A4", "file": "contacts.json", "parameters": {{}} }}

    Now, classify the task and generate the correct JSON response.
    """

    try:
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}]
            },
            verify=False,
        )
        llm_response = response.json()
        print(llm_response)
        # ✅ FIX: Ensure the response is valid JSON before parsing
        if "choices" not in llm_response or not llm_response["choices"]:
            raise ValueError("Invalid response from LLM: No choices found")

        raw_content = llm_response["choices"][0]["message"]["content"].strip()

        # ✅ FIX: Ensure the response content is valid JSON
        raw_content = raw_content.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            raise ValueError(f"LLM returned invalid JSON: {raw_content}")

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM API: {str(e)}")

# Task Execution Functions
def install_uv_and_run_script(email: str):
    subprocess.run(["pip", "install", "uv"], check=True)
    #subprocess.run(["python", "-m", "urllib.request", "-o", "datagen.py",
                    #"https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"])
    url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    output_file = "datagen.py"
    urllib.request.urlretrieve(url, output_file)
    print(f"File {output_file} has been downloaded successfully.")
    subprocess.run(["python", "datagen.py", email], check=True)

def format_markdown_file(file_path: str):
    print(Path(file_path))
    subprocess.run(["npx", "prettier@3.4.2", "--write", file_path], check=True,shell=True)

def count_days(file_path: str, output_file: str,day:str):
    """with open(file_path, "r") as f:
        dates = f.readlines()

    wednesday_count = sum(1 for date in dates if datetime.strptime(date.strip(), "%Y-%m-%d").weekday() == 2)

    with open(output_file, "w") as f:
        f.write(str(wednesday_count))"""
    l = []
    with open(file_path, "r") as file:
        s = file.readlines()
        for x in s:
            x = x.strip()
            formats = ["%Y/%m/%d %H:%M:%S", "%Y/%m/%d", "%d-%b-%Y", "%Y-%m-%d"]
            for fmt in formats:
                try:
                    date_obj = datetime.strptime(x, fmt)
                    l.append(date_obj.strftime("%A"))  # Return weekday name

                except ValueError:
                    continue  # Try the next format
    file.close()
    with open(f'C:/data/date-{day}s.txt', "a") as f:
        f.write(str(l.count(day)))

def sort_contacts(file_path: str, output_file: str):
    with open(file_path, "r") as f:
        contacts = json.load(f)

    sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))

    with open(output_file, "w") as f:
        json.dump(sorted_contacts, f, indent=4)

def extract_recent_logs(logs_dir: str, output_file: str):
    log_files = sorted(Path(logs_dir).glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:10]
    with open(output_file, "w") as f:
        for log_file in log_files:
            with open(log_file, "r") as lf:
                first_line = lf.readline().strip()
                f.write(first_line + "\n")

def extract_sender_email_from_email(file_path: str, output_file: str):
    """Extract the sender's email from the email content using LLM."""
    try:
        with open(file_path, "r") as f:
            email_content = f.read()

        # Prepare the prompt to pass the email content to LLM for extracting the sender's email
        prompt = f"""
        You are an AI assistant that extracts the sender's email address from email content.

        **Email Content:**
        {email_content}

        **Extract only the sender's email address.**

        **Output Format:**
        {{ "email": "<sender_email>" }}
        """

        # Pass the email content to the LLM for processing
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}]
            },
            verify=False,
        )
        llm_response = response.json()

        if "choices" not in llm_response or not llm_response["choices"]:
            raise ValueError("Invalid response from LLM: No choices found")

        raw_content = llm_response["choices"][0]["message"]["content"].strip()

        # ✅ FIX: Ensure the response content is valid JSON and parse the sender's email
        raw_content = raw_content.replace("```json", "").replace("```", "").strip()
        try:
            sender_info = json.loads(raw_content)
            sender_email = sender_info.get("email")
            if not sender_email:
                raise ValueError("No email address found in LLM response")

            # Save the extracted email address to the output file
            with open(output_file, "w") as out_f:
                out_f.write(sender_email)

            return f"Sender's email '{sender_email}' saved to {output_file}"

        except json.JSONDecodeError:
            raise ValueError(f"LLM returned invalid JSON: {raw_content}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting sender email: {str(e)}")

def extract_credit_card_from_image(image_path: str, output_file: str):
    try:
        # Step 1: Use OCR to extract text from the image
        pytesseract.pytesseract.tesseract_cmd=r'C:\Users\e430372\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)

        # Step 2: Prepare the prompt to pass the extracted text to LLM to extract the credit card number
        prompt = f"""
        You are an AI assistant tasked with extracting a credit card number from text.

        **Extracted Text from Image:**
        {extracted_text}

        **Task:**
        Find the credit card number in the text and return it without spaces or dashes.

        **Output Format:**
        {{ "credit_card_number": "<credit_card_number>" }}
        """

        # Step 3: Pass the extracted text to the LLM for processing
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}]
            },
            verify=False,
        )

        llm_response = response.json()

        if "choices" not in llm_response or not llm_response["choices"]:
            raise ValueError("Invalid response from LLM: No choices found")

        raw_content = llm_response["choices"][0]["message"]["content"].strip()

        # Step 4: Ensure the response content is valid JSON and extract the card number
        raw_content = raw_content.replace("```json", "").replace("```", "").strip()
        try:
            card_info = json.loads(raw_content)
            card_number = card_info.get("credit_card_number")
            if not card_number:
                raise ValueError("No credit card number found in LLM response")

            # Step 5: Save the extracted credit card number to the output file
            with open(output_file, "w") as out_f:
                out_f.write(card_number)

            return f"Credit card number '{card_number}' saved to {output_file}"

        except json.JSONDecodeError:
            raise ValueError(f"LLM returned invalid JSON: {raw_content}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting credit card number: {str(e)}")

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

