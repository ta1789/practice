import os
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
import csv
import markdown
import duckdb
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import numpy as np
import urllib
#from moviepy.editor import AudioFileClip

# API Token for LLM
load_dotenv()
TOKEN = os.getenv("API_PROXY")
if not TOKEN:
    raise ValueError("API_PROXY token not found in environment variables.")

# Initialize FastAPI app
app = FastAPI()

# Root data directory
DATA_DIR = Path("C:/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists


class Task(BaseModel):
    description: str


def validate_file_path(file_path: str) -> bool:
    """Ensure the file is within the allowed directory."""
    abs_file_path = os.path.abspath(file_path)
    abs_allowed_dir = os.path.abspath(DATA_DIR)
    return abs_file_path.startswith(abs_allowed_dir)


def prevent_deletion(file_path: str) -> bool:
    """Ensure no file deletion operation is allowed."""
    return "delete" in file_path.lower() or "remove" in file_path.lower()


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
    - A9: Extracting list of comments
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
        response.raise_for_status()  # Raise an error for bad responses
        llm_response = response.json()

        if "choices" not in llm_response or not llm_response["choices"]:
            raise ValueError("Invalid response from LLM: No choices found")

        raw_content = llm_response["choices"][0]["message"]["content"].strip()
        raw_content = raw_content.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_content)

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM API: {str(e)}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON.")


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

def calculate_gold_ticket_sales(db_path: str, output_file: str):
    """Calculate the total sales for the 'Gold' ticket type."""
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # SQL query to calculate total sales for 'Gold' ticket type
        query = """
        SELECT SUM(units * price) FROM tickets WHERE type = 'Gold';
        """
        cursor.execute(query)

        # Fetch the result
        total_sales = cursor.fetchone()[0]  # This returns the sum of sales, or None if no result

        if total_sales is None:
            total_sales = 0  # If no result is found, set it to 0

        conn.close()

        # Write the result to the output file
        with open(output_file, "w") as f:
            f.write(f"Total Sales for Gold Tickets: {total_sales}\n")

        return f"Total sales for Gold tickets calculated and saved to {output_file}"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating Gold ticket sales: {str(e)}")

# Task Execution Functions
def create_index_for_markdown_files():
    docs_dir = Path("/data/docs")
    index = {}

    # Iterate through all markdown files in the /data/docs/ directory and its subdirectories
    for md_file in docs_dir.rglob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find the first H1 (line starting with '#')
        title = None
        for line in lines:
            if line.startswith("# "):  # H1 header starts with "# "
                title = line.strip("# ").strip()  # Remove the '#' and any leading/trailing whitespace
                break

        # If we found a title, add it to the index
        if title:
            # Use the filename without the /data/docs/ prefix
            file_name = md_file.relative_to(docs_dir).as_posix()
            index[file_name] = title

    # Write the index to /data/docs/index.json
    with open(docs_dir / "index.json", "w", encoding="utf-8") as index_file:
        json.dump(index, index_file, indent=4, ensure_ascii=False)

    return "Index created successfully at /data/docs/index.json"

def get_embeddings(texts: list) -> np.ndarray:
    """Uses the LLM API to generate embeddings for the list of texts."""
    try:
        # Prepare the request to fetch embeddings from the API
        response = requests.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": f"Generate embeddings for the following texts: {texts}"}]
            },
            verify=False,
        )

        embeddings = response.json()

        # Debugging: Print the response to inspect the structure
        print("Embedding Response:", embeddings)

        if "choices" not in embeddings or not embeddings["choices"]:
            raise ValueError("Invalid response from LLM: No embeddings found")

        # Extract the embeddings from the response
        embedding_str = embeddings['choices'][0]['message']['content']

        embeddings_list = []

        for text in texts:
            try:
                # Print the current text to help identify where things go wrong
                print(f"Processing text: {text}")

                if text in embedding_str:
                    # If the text is in the response, process its embedding
                    embedding_str = embedding_str.split(text)[1]
                    embedding_list = [float(x) for x in embedding_str.split()]
                    embeddings_list.append(embedding_list)
                else:
                    # If the text is not found, log a message
                    print(f"Text '{text}' not found in embedding response.")
                    embeddings_list.append([])  # Add empty list for this text, so the process doesn't break.

            except ValueError as ve:
                print(f"Error processing embedding for text: {text}")
                print(f"Details: {ve}")
                raise ValueError(f"Error processing embedding for text: {text}")

        print(f"Generated {len(embeddings_list)} embeddings.")

        return np.array(embeddings_list)

    except requests.RequestException as e:
        raise Exception(f"Error fetching embeddings: {str(e)}")
    except ValueError as ve:
        raise Exception(f"Error processing embeddings: {str(ve)}")


# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_most_similar_comments(file_path: str, output_file: str):
    try:
        # Step 1: Read the comments from the file
        with open(file_path, "r") as f:
            comments = f.readlines()

        # Debugging: Print the number of comments
        print(f"Read {len(comments)} comments.")

        # Step 2: Generate embeddings for the comments
        embeddings = get_embeddings(comments)

        # Debugging: Print the number of embeddings
        print(f"Number of embeddings: {len(embeddings)}")

        # Step 3: Check if the length of embeddings matches the number of comments
        if len(embeddings) != len(comments):
            raise ValueError("The number of embeddings does not match the number of comments.")

        # Step 4: Compute pairwise similarities
        max_similarity = -1
        most_similar_pair = ("", "")
        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                # Convert each embedding to a NumPy array for similarity calculation
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_pair = (comments[i].strip(), comments[j].strip())

        # Step 5: Write the most similar pair to the output file
        with open(output_file, "w") as f:
            f.write(f"Most Similar Pair:\nComment 1: {most_similar_pair[0]}\nComment 2: {most_similar_pair[1]}\n")

        return f"Most similar comments written to {output_file}"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding most similar comments: {str(e)}")

@app.get("/run")
def run_task(task: str = Query(..., description="Task description")):
    try:
        task_info = parse_task_with_llm(task)
        operation = task_info.get("operation")
        file_path = task_info.get("file", "")
        params = task_info.get("parameters", {})
        print(task_info)

        if operation in ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"]:
            if file_path and not validate_file_path(file_path):
                raise HTTPException(status_code=403, detail="Access to this file is forbidden.")
            if prevent_deletion(task):
                raise HTTPException(status_code=403, detail="File deletion is not allowed.")

            task_mapping = {
                "B3": lambda: fetch_data_and_save(params.get("url", ""), params.get("output_file", "/data/data.json")),
                "B4": lambda: clone_and_commit_git_repo(params.get("repo_url", ""), params.get("commit_message", "")),
                "B5": lambda: run_sql_query(params.get("db_path", "database.db"), params.get("query", "")),
                "B6": lambda: scrape_website(params.get("url", ""), params.get("output_file", "/data/scraped_data.json")),
                "B7": lambda: compress_or_resize_image(file_path, params.get("output", "/data/resized_image.jpg")),
                "B9": lambda: convert_markdown_to_html(file_path, params.get("output_file", "/data/output.html")),
                "B10": lambda: filter_csv_and_return_json(file_path, params.get("filter_column", ""), params.get("filter_value", "")),
            }

            if operation in task_mapping:
                result = task_mapping[operation]()
                return {"message": f"Task {operation} executed successfully.", "result": result or "Success"}

            raise HTTPException(status_code=400, detail=f"Invalid task operation: {operation}")

        else:
            task_mapping = {
                "A1": lambda: install_uv_and_run_script(params.get("email", "default@example.com")),
                "A2": lambda: format_markdown_file(file_path),
                "A3": lambda: count_days(file_path, params.get("output_file", "/data/output.txt"), params.get("day", "monday")),
                "A4": lambda: sort_contacts(file_path, params.get("output_file", "/data/sorted_contacts.json")),
                "A5": lambda: extract_recent_logs(file_path, params.get("output_file", "/data/logs-recent.txt")),
                "A6": lambda: create_index_for_markdown_files(),
                "A7": lambda: extract_sender_email_from_email(file_path, params.get("output_file", "/data/email-sender.txt")),
                "A8": lambda: extract_credit_card_from_image(file_path, params.get("output_file", "/data/credit-card.txt")),
                "A9": lambda: find_most_similar_comments(params.get("file_path", "C:/data/comments.txt"), params.get("output_file", "C:/data/comments-similar.txt")),
                "A10": lambda: calculate_gold_ticket_sales(params.get("db_path", "C:/data/ticket-sales.db"), params.get("output", "C:/data/ticket-sales-gold.txt")),
            }

            if operation in task_mapping:
                result = task_mapping[operation]()
                return {"message": f"Task {operation} executed successfully.", "result": result or "Success"}

            raise HTTPException(status_code=400, detail=f"Invalid task operation: {operation}")

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/read")
def read_file(path: str = Query(..., description="File path to read")):
    try:
        # Convert path to absolute path and normalize it
        abs_path = os.path.abspath(path)

        # Validate that the file is within DATA_DIR (B1)
        if not validate_file_path(path):
            raise HTTPException(status_code=403,detail="Access denied. Can only read files from the data directory.")

        # Check if file exists
        if not os.path.exists(abs_path):
            raise HTTPException(status_code=404,detail="File not found")

        # Check if path is a file (not a directory)
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=400,detail="Path must point to a file")

        # Read and return file contents
        try:
            with open(abs_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return content
        except UnicodeDecodeError:
            # Try reading as binary if text reading fails
            with open(abs_path, 'rb') as file:
                content = file.read()
                return Response(content=content,media_type="application/octet-stream")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Error reading file: {str(e)}")


# Task Implementations for B3 - B10

# B3. Fetch data from an API and save it
def fetch_data_and_save(url: str, output_file: str):
    try:
        response = requests.get(url,verify=False)
        response.raise_for_status()  # Check for errors
        with open(output_file, "w") as f:
            json.dump(response.json(), f, indent=4)
        return f"Data saved to {output_file}"
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")


# B4. Clone a git repository and make a commit
def clone_and_commit_git_repo(repo_url: str, commit_message: str):
    try:
        repo_dir = Path("C:/data/repo")
        if repo_dir.exists():
            repo = git.Repo(repo_dir)
        else:
            repo = git.Repo.clone_from(repo_url, repo_dir)

        # Make a commit
        repo.git.add(A=True)
        repo.index.commit(commit_message)
        repo.remote().push()
        return f"Repository cloned and commit '{commit_message}' made."
    except git.exc.GitError as e:
        raise HTTPException(status_code=500, detail=f"Git error: {str(e)}")


# B5. Run a SQL query on a SQLite or DuckDB database
def run_sql_query(db_path: str, query: str):
    try:
        if db_path.endswith(".db"):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
        elif db_path.endswith(".duckdb"):
            conn = duckdb.connect(db_path)
            cursor = conn.cursor()
        else:
            raise ValueError("Unsupported database format.")

        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running SQL query: {str(e)}")


def scrape_website(url: str, output_file: str):
    try:
        response = requests.get(url,verify=False)
        soup = BeautifulSoup(response.content, "html.parser")
        with open(output_file, "w") as f:
            json.dump({"content": soup.get_text()}, f, indent=4)
        return f"Data scraped and saved to {output_file}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping website: {str(e)}")


# B7. Compress or resize an image
def compress_or_resize_image(file_path: str, output_file: str):
    try:
        img = Image.open(file_path)
        img = img.resize((img.width // 2, img.height // 2))
        img.save(output_file, optimize=True, quality=85)
        return f"Image saved as {output_file}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resizing image: {str(e)}")



# B9. Convert Markdown to HTML
def convert_markdown_to_html(file_path: str, output_file: str):
    try:
        with open(file_path, "r") as f:
            markdown_text = f.read()
        html = markdown.markdown(markdown_text)
        with open(output_file, "w") as f:
            f.write(html)
        return f"Markdown converted to HTML and saved as {output_file}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting Markdown to HTML: {str(e)}")


# B10. Filter a CSV file and return JSON data
def filter_csv_and_return_json(file_path: str, filter_column: str, filter_value: str):
    try:
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            filtered_data = [row for row in reader if row.get(filter_column) == filter_value]
        return filtered_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error filtering CSV: {str(e)}")


# Run FastAPI server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)