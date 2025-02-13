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

# API Token for LLM
load_dotenv()
TOKEN = os.getenv("API_PROXY")
print(TOKEN)

# Initialize FastAPI app
app = FastAPI()

# Root data directory
DATA_DIR = Path("C:/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

class task(BaseModel):
    description: str

def parse_task_with_llm(task_description: str) -> dict:
    """Use LLM to classify the task and extract execution details."""
    prompt = f"""
    You are an AI assistant that converts task descriptions into structured JSON.

    **Task List:**
    - A1: Install uv & run datagen.py (requires email)
    - A2: Format a Markdown file using Prettier
    - A3: Count Wednesdays in a date list file
    - A4: Sort contacts from a JSON file
    - A5: Extract recent log file lines
    - A6: Create an index.json from Markdown files
    - A7: Extract sender email from an email text file
    - A8: Extract a credit card number from an image
    - A9: Extract transactions above $1000 from CSV
    - A10: Get a weather forecast for a city

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
    subprocess.run(["python", "-m", "urllib.request", "-o", "datagen.py",
                    "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"])
    subprocess.run(["python", "datagen.py", email], check=True)

def format_markdown_file(file_path: str):
    subprocess.run(["npx", "prettier@3.4.2", "--write", file_path], check=True)

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

@app.get("/run")
def run_task(task: str = Query(..., description="Task description")):
    """Execute a task based on LLM parsing"""
    try:
        print("Received Request:", task)
        task_info = parse_task_with_llm(task)
        print("Parsed Task:", task_info)

        operation = task_info.get("operation")
        file_path = task_info.get("file", "")
        params = task_info.get("parameters", {})

        task_mapping = {
            "A1": lambda: install_uv_and_run_script(params.get("email", "default@example.com")),
            "A2": lambda: format_markdown_file(file_path),
            "A3": lambda: count_days(file_path, params.get("output", "output.txt"),params.get("day", "monday")),
            "A4": lambda: sort_contacts(file_path, params.get("output", "sorted_contacts.json")),
            "A5": lambda: extract_recent_logs(file_path, params.get("output", "logs.txt")),
        }

        if operation in task_mapping:
            result = task_mapping[operation]()
            return {"message": f"Task {operation} executed successfully.", "result": result or "Success"}

        raise HTTPException(status_code=400, detail=f"Invalid task operation: {operation}")

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
