import os
import re
import json

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import shutil
from typing import Dict, List
import fitz
import PyPDF2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import pymupdf4llm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TRANSFORMERS_CACHE"] = r'C:\\Windows\\system32\\config\\systemprofile\\.cache\\huggingface\\hub'
os.environ['HF_HOME'] = r'C:\\Windows\\system32\\config\\systemprofile\\.cache\\huggingface\\hub'

from a2wsgi import ASGIMiddleware
app = FastAPI()
wsgi_app = ASGIMiddleware(app)


from difflib import SequenceMatcher

# Load the Sentence Transformer models
from sentence_transformers import SentenceTransformer

model1 = SentenceTransformer('all-MiniLM-L6-v2')
model2 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify allowed origins here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preset department keywords
department_keywords = {
    'Analytical Research and Development': ['ar'],
    'Engineering': ['se', 'eg', 'en', 'engineering'],
    'IT': ['it', 'information technology'],
    'Temperature Mapping': ['tm', 'temperature mapping'],
    'Production': ['TB', 'SH', 'production'],
    'Quality Assurance (QA)': ['qa', 'quality assurance'],
    'Quality Control (QC)': ['qc', 'hq', 'hm', 'quality control'],
    'Requalification Protocol': ['rq', 'requalification protocol'],
}

class CategorizeRequest(BaseModel):
    main_folder: str

destination_base = r""
heatmap_base = r""
HIGHLIGHTED_DIR = r""

def generate_tree(directory, level=1, max_level=2):
    dir_tree = {}
    items = sorted(os.listdir(directory))

    # Use parallel processing for directories
    def process_item(item):
        path = os.path.join(directory, item)
        if os.path.isdir(path):
            if level < max_level:
                return (item, generate_tree(path, level + 1, max_level))
            else:
                return (item, {})
        else:
            return (item, None)

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_item, items)

    for item, subtree in results:
        dir_tree[item] = subtree

    return dir_tree


def categorize_company_folder(company_folder_path):
    pdf_files = [f for f in os.listdir(company_folder_path) if f.endswith('.pdf')]

    if not pdf_files:
        return company_folder_path.split(os.sep)[-1], {}  # Return folder name and empty dict if no PDFs

    categorized_pdfs = {dept: [] for dept in department_keywords.keys()}
    categorized_pdfs['Uncategorized'] = []

    # Pre-compile department keyword patterns for faster matching
    keyword_patterns = {dept: re.compile('|'.join(map(re.escape, keywords)), re.IGNORECASE)
                        for dept, keywords in department_keywords.items()}

    def categorize_file(pdf_file):
        best_match = None
        best_position = len(pdf_file)

        # Search for keywords in parallel using compiled regex patterns
        for department, pattern in keyword_patterns.items():
            match = pattern.search(pdf_file)
            if match and match.start() < best_position:
                best_match = department
                best_position = match.start()

        return best_match if best_match else 'Uncategorized', pdf_file

    # Use ThreadPoolExecutor to categorize files in parallel
    with ThreadPoolExecutor() as executor:
        categorized_results = executor.map(categorize_file, pdf_files)

    # Populate categorized PDFs
    for department, pdf_file in categorized_results:
        categorized_pdfs[department].append(pdf_file)

    return company_folder_path.split(os.sep)[-1], categorized_pdfs


@app.post("/categorize/")
def categorize_pdfs(request: CategorizeRequest):
    main_folder = request.main_folder

    # Check if main folder exists
    if not os.path.exists(main_folder):
        raise HTTPException(status_code=404, detail=f"Main folder '{main_folder}' does not exist.")

    with ThreadPoolExecutor() as executor:
        future_to_company = {executor.submit(categorize_company_folder, os.path.join(main_folder, company_folder)): company_folder for company_folder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, company_folder))}

        for future in as_completed(future_to_company):
            company_folder, categorized_pdfs = future.result()
            print(f"Processing company folder: {company_folder}")

            # Move files into the company's department-specific sub-folders
            for department, files in categorized_pdfs.items():
                if files:
                    destination_folder = os.path.join(destination_base, company_folder, department)
                    os.makedirs(destination_folder, exist_ok=True)

                    for file in files:
                        source_path = os.path.join(os.path.join(main_folder, company_folder), file)
                        destination_path = os.path.join(destination_folder, file)

                        try:
                            shutil.copy(source_path, destination_path)  # Move the file
                            print(f"Moved '{file}' from '{company_folder}' to '{destination_folder}'")
                        except Exception as e:
                            print(f"Error moving '{file}': {e}")

    # Generate the JSON structure of the destination folder
    destination_structure = generate_tree(destination_base)

    # Return the folder structure as JSON after the categorization
    return {
        "destination_structure": json.dumps(destination_structure, indent=4)
    }



# ------------------------------------------------------------------------------------------------------------------------------
# In_Folder_Wise_Similarity
# ------------------------------------------------------------------------------------------------------------------------------


# Optimized function to calculate similarity
def calculate_similarity(texts, weights):
    # Step 1: Optimize TF-IDF vectorization by using sparse matrices
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    tfidf_similarity = cosine_similarity(tfidf_matrix)

    # Step 2: Use parallel processing to compute embeddings
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            'model1': executor.submit(model1.encode, texts),
            'model2': executor.submit(model2.encode, texts)
        }

        embeddings_model1 = futures['model1'].result()
        embeddings_model2 = futures['model2'].result()

    # Step 3: Calculate cosine similarities for embeddings
    cosine_sim_model1 = cosine_similarity(embeddings_model1)
    cosine_sim_model2 = cosine_similarity(embeddings_model2)

    # Step 4: Combine the similarity results with the given weights
    combined_similarity = (
        weights['tfidf'] * tfidf_similarity +
        weights['model1'] * cosine_sim_model1 +
        weights['model2'] * cosine_sim_model2
    )

    return combined_similarity


# ------------------------------------------------------------------------------------------------------------------------------
# Sub_Folder_Wise_Section_Similarity
# ------------------------------------------------------------------------------------------------------------------------------

# Define request body for uploading folder
class UploadFolderRequest(BaseModel):
    department: str
    threshold: float = 0.85  # Default threshold
    companies: List[str]


# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Function to convert PDF to Markdown
def convert_pdf_to_markdown(file_path):
    return pymupdf4llm.to_markdown(file_path)

# Parallelize the extraction of PDF text using ThreadPoolExecutor
def extract_text_from_multiple_pdfs(file_paths):
    texts = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(extract_text_from_pdf, path): path for path in file_paths}

        for future in as_completed(futures):
            file_path = futures[future]
            try:
                text = future.result()
                if text:
                    texts.append((file_path, text))
            except Exception as e:
                print(f"Error extracting {file_path}: {e}")
    return texts

# Parallelize the conversion of PDFs to Markdown
def convert_pdfs_to_markdown(file_paths):
    markdowns = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(convert_pdf_to_markdown, path): path for path in file_paths}

        for future in as_completed(futures):
            file_path = futures[future]
            try:
                markdown = future.result()
                markdowns.append((file_path, markdown))
            except Exception as e:
                print(f"Error converting {file_path} to Markdown: {e}")
    return markdowns


def extract_unique_sections(markdown_text):
    # Pattern to match sections ending in .0
    section_pattern = r'(\*\*(\d+\.0)\*\*)\s+(\*\*.*?\*\*)\n\n(.*?)(?=\n\n\*\*\d+\.\d+\*\*|$)'
    sections_data = re.findall(section_pattern, markdown_text, re.DOTALL)

    unique_sections = {}
    seen_identifiers = set()
    seen_content = set()

    for full_section_num, identifier, section_title, content in sections_data:
        # Skip if the section identifier has already been processed
        if identifier not in seen_identifiers:
            normalized_content = content.strip().lower()
            if normalized_content not in seen_content:
                section_id = f"{full_section_num} {section_title.strip()}"
                unique_sections[section_id] = content.strip()
                seen_identifiers.add(identifier)  # Mark this identifier as processed
                seen_content.add(normalized_content)

    return unique_sections


# Function to compare sections using TF-IDF, ignoring section numbers
def compare_sections(sections1, sections2):
    similarities = {}
    common_sections = set(sections1.keys()).intersection(set(sections2.keys()))

    for section in common_sections:
        content1 = sections1[section]
        content2 = sections2[section]

        # Compute TF-IDF Cosine Similarity for section content
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([content1, content2])
        cosine_sim_score = cosine_similarity(tfidf_matrix)[0][1]

        similarities[section] = round(cosine_sim_score, 2)

    return similarities


@app.post("/in-folder-compare/")
def upload_folder(request: UploadFolderRequest):
    company = request.companies[0]
    department = request.department
    threshold = request.threshold

    folder_path = os.path.join(destination_base, company, department)
    print(folder_path)
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="Invalid folder path provided.")

    texts = []
    files_list = []
    section_data = {}

    # Step 1: Extract text from PDF files in the given folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(folder_path, file_name)
            text = extract_text_from_pdf(file_path)
            if text:  # Only append non-empty texts
                texts.append(text)
                files_list.append(file_name)
            else:
                logger.warning(f"No text extracted from {file_name}")

    results = []
    markdown_cache = {}
    doc_details = {}
    doc_labels = [f"doc{i+1}" for i in range(len(files_list))]

    if texts:
        weights = {'tfidf': 0.25, 'model1': 0.35, 'model2': 0.40}
        similarity_matrix = calculate_similarity(texts, weights)

        # Map doc labels to original file names
        doc_details = {label: name for label, name in zip(doc_labels, files_list)}

        # Generate and save the heatmap with doc labels
        plt.figure(figsize=(10, 8))  # Adjust figure size
        sns.heatmap(similarity_matrix, xticklabels=doc_labels, yticklabels=doc_labels, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Document Similarity Heatmap for {company} - {department}")
        plt.xticks(rotation = 45, ha = 'right')  # Rotate x-axis labels for readability
        plt.tight_layout()  # Adjust layout to prevent clipping
        heatmap_path = os.path.join(heatmap_base, "heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        logger.info(f"Heatmap saved at {heatmap_path}")

        # Step 2: Identify pairs with similarity above the threshold
        for i in range(len(files_list)):
            for j in range(i + 1, len(files_list)):
                if similarity_matrix[i][j] > threshold:
                    pair_key = f"{files_list[i]} <-> {files_list[j]}"

                    comparison_result = {
                        "file1": files_list[i],
                        "file2": files_list[j],
                        "similarity_score": round(similarity_matrix[i][j], 2)
                    }

                    logger.info(f"Similar documents found: {pair_key} with similarity {similarity_matrix[i][j]:.2f}")

                    file1, file2 = pair_key.split(" <-> ")
                    file1_path = os.path.join(folder_path, file1)
                    file2_path = os.path.join(folder_path, file2)

                    # Convert file1 to Markdown if not already converted
                    if file1 not in markdown_cache:
                        markdown_cache[file1] = convert_pdf_to_markdown(file1_path)
                    # Convert file2 to Markdown if not already converted
                    if file2 not in markdown_cache:
                        markdown_cache[file2] = convert_pdf_to_markdown(file2_path)

                    # Extract unique sections and clean titles
                    def clean_title(title):
                        # Remove leading numbers, colons, dashes, and asterisks
                        return re.sub(r'^\d+(\.\d+)?\s*|[:\-]', '', title.replace('**', '').strip()).strip()

                    sections1 = {clean_title(sec): content for sec, content in extract_unique_sections(markdown_cache[file1]).items()}
                    sections2 = {clean_title(sec): content for sec, content in extract_unique_sections(markdown_cache[file2]).items()}
                    section_similarities_result = compare_sections(sections1, sections2)

                    # Store section similarities after cleaning titles
                    if len(section_similarities_result) != 0:
                        cleaned_section_similarities = {
                            clean_title(key): value for key, value in section_similarities_result.items()
                        }
                        comparison_result["section_similarities"] = cleaned_section_similarities

                    results.append(comparison_result)

                    # Identify uncommon section titles with implied match handling
                    def is_similar_overlay(title1, title2, threshold=0.8):
                        return SequenceMatcher(None, title1, title2).ratio() > threshold

                    def find_uncommon_sections(sections1, sections2, threshold = 0.8):
                        uncommon_file1 = []
                        uncommon_file2 = list(sections2)  # Start with all sections in file2 as uncommon

                        for title1 in sections1:
                            found_match = False
                            for title2 in sections2:
                                if is_similar_overlay(title1, title2, threshold):
                                    found_match = True
                                    if title2 in uncommon_file2:
                                        uncommon_file2.remove(title2)
                                    break
                            if not found_match:
                                uncommon_file1.append(title1)

                        return {"file1": uncommon_file1, "file2": uncommon_file2}

                    # Get the uncommon section titles
                    uncommon_titles = find_uncommon_sections(list(sections1.keys()), list(sections2.keys()))
                    comparison_result["uncommon_sections"] = uncommon_titles


    # Return results with original file names and document details with doc1, doc2 mappings
    return JSONResponse(content={
        "results": {company: results},
        "doc_details": doc_details
    })

# ------------------------------------------------------------------------------------------------------------------------------
# Out_Folder_Wise_Similarity
# ------------------------------------------------------------------------------------------------------------------------------
# Preset weights and similarity threshold
weights = {'tfidf': 0.25, 'model1': 0.35, 'model2': 0.40}

# Define input model
class ComparisonRequest(BaseModel):
    company_list: List[str]
    department_choice: str

import os
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Caching extracted text for faster access
text_cache = {}

# Function to extract text and cache the result
def extract_text_cached(file_path):
    if file_path in text_cache:
        return text_cache[file_path]

    text = extract_text_from_pdf(file_path)
    text_cache[file_path] = text
    return text

# Function to compare two files
def compare_files(file1_path, file2_path):
    text1 = extract_text_cached(file1_path)
    text2 = extract_text_cached(file2_path)

    if not text1 or not text2:
        return 0  # Return a score of 0 if any of the files could not be read

    similarity_matrix = calculate_similarity([text1, text2])
    return similarity_matrix[0][1]  # Return the similarity score between the two files

# Parallelized function for comparing files across companies
def compare_department_across_companies(main_folder_path, company_list, department_choice, similarity_threshold):
    comparison_results = []

    def process_company_pair(company1, company2):
        company1_path = os.path.join(main_folder_path, company1, department_choice)
        company2_path = os.path.join(main_folder_path, company2, department_choice)

        if not os.path.isdir(company1_path) or not os.path.isdir(company2_path):
            return []

        company1_files = [f for f in os.listdir(company1_path) if f.endswith('.pdf')]
        company2_files = [f for f in os.listdir(company2_path) if f.endswith('.pdf')]

        results = []
        for file1 in company1_files:
            file1_path = os.path.join(company1_path, file1)
            for file2 in company2_files:
                file2_path = os.path.join(company2_path, file2)
                score = compare_files(file1_path, file2_path)

                if score > similarity_threshold:
                    result = {
                        "department": department_choice,
                        "company1": company1,
                        "file1": file1,
                        "company2": company2,
                        "file2": file2,
                        "similarity_score": score
                    }
                    results.append(result)

        return results

    # Parallelize across company pairs
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, company1 in enumerate(company_list):
            for company2 in company_list[i+1:]:
                futures.append(executor.submit(process_company_pair, company1, company2))

        # Collect the results from all futures
        for future in futures:
            comparison_results.extend(future.result())

    return comparison_results


class CompareFoldersRequest(BaseModel):
    companies: List[str]
    department: str
    threshold: float = 0.85

@app.post("/out-folder-compare/")
def compare_across_folders(request: CompareFoldersRequest):
    companies = request.companies
    department = request.department
    threshold = request.threshold

    if not os.path.exists(destination_base):
        raise HTTPException(status_code = 400, detail = "Invalid main folder path provided.")

    # Step 1: Prepare file paths for PDFs from the chosen companies and department
    company_files = {}

    for company in companies:
        company_folder = os.path.join(destination_base, company, department)

        if not os.path.exists(company_folder):
            raise HTTPException(status_code=400, detail=f"Folder for {company} not found.")

        company_files[company] = [
            os.path.join(company_folder, file_name) for file_name in os.listdir(company_folder) if file_name.endswith('.pdf')
        ]

    # Step 2: Extract text and compute document similarity across companies
    results = []
    company_pairs = [(companies[i], companies[j]) for i in range(len(companies)) for j in range(i + 1, len(companies))]

    # Caching for markdown and text to avoid redundant operations
    text_cache = {}
    markdown_cache = {}

    for company1, company2 in company_pairs:
        for file1 in company_files[company1]:
            if file1 not in text_cache:
                text_cache[file1] = extract_text_from_pdf(file1)
            text1 = text_cache[file1]

            for file2 in company_files[company2]:
                if file2 not in text_cache:
                    text_cache[file2] = extract_text_from_pdf(file2)
                text2 = text_cache[file2]

                # Step 3: Calculate document similarity once per pair
                similarity_score = calculate_pdf_similarity(text1, text2)

                if similarity_score > threshold:
                    comparison_result = {
                        "department": department,
                        "company1": f"{company1} || {company2}",
                        "file1": os.path.basename(file1),
                        "file2": os.path.basename(file2),
                        "similarity_score": int(similarity_score * 100),
                    }

                    # Step 4: Convert PDFs to markdown (cached) and compare sections
                    if file1 not in markdown_cache:
                        markdown_cache[file1] = convert_pdf_to_markdown(file1)
                    markdown1 = markdown_cache[file1]

                    if file2 not in markdown_cache:
                        markdown_cache[file2] = convert_pdf_to_markdown(file2)
                    markdown2 = markdown_cache[file2]

                    # Extract and clean unique sections from each file
                    def clean_section_name(section):
                        # Remove leading numbers, colons, dashes, asterisks, and convert to uppercase
                        return re.sub(r'^\d+(\.\d+)?\s*|[:\-]', '', section.replace('**', '').strip()).strip().upper()

                    sections1 = {clean_section_name(sec): content for sec, content in extract_unique_sections(markdown1).items()}
                    sections2 = {clean_section_name(sec): content for sec, content in extract_unique_sections(markdown2).items()}
                    section_similarities_result = compare_sections(sections1, sections2)

                    # Store section similarities after cleaning titles
                    if len(section_similarities_result) != 0:
                        cleaned_section_similarities = {
                            clean_section_name(key): value for key, value in section_similarities_result.items()
                        }
                        comparison_result["section_similarities"] = cleaned_section_similarities

                    # Append the result to the list
                    results.append(comparison_result)

                    # Implied match handling for uncommon sections
                    def is_similar_overlay(title1, title2, threshold=0.8):
                        return SequenceMatcher(None, title1, title2).ratio() > threshold

                    def find_uncommon_sections(sections1, sections2, threshold=0.8):
                        uncommon_file1 = []
                        uncommon_file2 = list(sections2)  # Start with all sections in file2 as uncommon

                        for title1 in sections1:
                            found_match = False
                            for title2 in sections2:
                                if is_similar_overlay(title1, title2, threshold):
                                    found_match = True
                                    if title2 in uncommon_file2:
                                        uncommon_file2.remove(title2)
                                    break
                            if not found_match:
                                uncommon_file1.append(title1)

                        return {"file1": uncommon_file1, "file2": uncommon_file2}

                    # Get uncommon section titles with implied match handling
                    uncommon_titles = find_uncommon_sections(list(sections1.keys()), list(sections2.keys()))
                    comparison_result["uncommon_sections"] = uncommon_titles


    return JSONResponse(content={"comparisons": results})


# Helper function to clean section names
def clean_section_name(section_name):
    # Remove leading numbers, dots, asterisks, and spaces, as well as trailing asterisks
    cleaned = re.sub(r'^[\d\.\*\s]+', '', section_name)  # Remove leading numbers, dots, and asterisks
    cleaned = re.sub(r'\*+$', '', cleaned)  # Remove trailing asterisks
    return cleaned.strip()  # Strip extra spaces


# Function to calculate the similarity between PDF texts using TF-IDF or other models
def calculate_pdf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim_score = cosine_similarity(tfidf_matrix)[0][1]
    return cosine_sim_score


# ------------------------------------------------------------------------------------------------------------------------------
# PDF_Overlay
# -----------------------------------------------------------------------------------------------------------------------------

class PDFPaths(BaseModel):
    pdf1_path: str
    pdf2_path: str


def convert_pdf_to_markdown_overlay(file_path):
    return pymupdf4llm.to_markdown(file_path)

def convert_pdfs_to_markdown_overlay(file_paths):
    markdowns = []
    with ThreadPoolExecutor(max_workers = os.cpu_count()) as executor:
        futures = {executor.submit(convert_pdf_to_markdown_overlay, path): path for path in file_paths}

        for future in as_completed(futures):
            file_path = futures[future]
            try:
                markdown = future.result()
                markdowns.append((file_path, markdown))
            except Exception as e:
                print(f"Error converting {file_path} to Markdown: {e}")
    return markdowns


def extract_unique_sections_overlay(markdown_text):
    section_pattern = r'(\*\*\d+[.,][0-1]\*\*)\s+(\*\*.*?\*\*)\n\n(.*?)(?=\n\n\*\*\d+[.,][0-1]\*\*|$)'
    sections_data = re.findall(section_pattern, markdown_text, re.DOTALL)

    unique_sections = {}
    seen_content = set()

    for section_num, section_title, content in sections_data:
        # Check if the content has a long sequence of dots, indicating it should be excluded
        if '.....' in content:
            continue

        normalized_content = content.strip().lower()
        if normalized_content not in seen_content:
            section_id = f"{section_num} {section_title.strip()}"
            unique_sections[section_id] = content.strip()
            seen_content.add(normalized_content)

    return unique_sections


# Function to compare sections using TF-IDF, ignoring section numbers
def compare_sections_overlay(sections1, sections2):
    similarities = {}
    common_sections = set(sections1.keys()).intersection(set(sections2.keys()))

    for section in common_sections:
        content1 = sections1[section]
        content2 = sections2[section]

        # Compute TF-IDF Cosine Similarity for section content
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([content1, content2])
        cosine_sim_score = cosine_similarity(tfidf_matrix)[0][1]

        similarities[section] = round(cosine_sim_score, 2)

    return similarities

# Extract Sections and Clean Titles
def clean_title_overlay(title):
    """Remove leading numbers, colons, dashes, and asterisks."""
    return re.sub(r'^\d+(\.\d+)?\s*|[:\-]', '', title.replace('**', '').strip()).strip()


# Find Uncommon Sections
def is_similar_overlay(title1, title2, threshold=0.8):
    """Check if two titles are similar based on a threshold."""
    return SequenceMatcher(None, title1, title2).ratio() > threshold

def find_uncommon_sections_overlay(sections1, sections2, threshold=0.8):
    """Find sections unique to each PDF."""
    uncommon_file1 = []
    uncommon_file2 = list(sections2)  # Start with all sections in file2 as uncommon

    for title1 in sections1:
        found_match = False
        for title2 in sections2:
            if is_similar_overlay(title1, title2, threshold):
                found_match = True
                if title2 in uncommon_file2:
                    uncommon_file2.remove(title2)
                break
        if not found_match:
            uncommon_file1.append(title1)

    return {"file1": uncommon_file1, "file2": uncommon_file2}


def extract_sections_overlay(sections, similarity_result, threshold=0.05):
    """Extract sections that are above the similarity threshold."""
    return {key: value for key, value in sections.items() if key in similarity_result and similarity_result[key] > threshold}

def clean_text_overlay(text):
    """Remove special characters from the text, such as asterisks."""
    return text.replace('*', '').strip()

def highlight_sections_and_titles_in_pdf_overlay(pdf_path, sections_to_highlight, titles_to_highlight):
    """Highlight specified sections in green and titles in red within the same PDF."""
    os.makedirs(HIGHLIGHTED_DIR, exist_ok=True)
    highlighted_file_path = os.path.join(
        HIGHLIGHTED_DIR,
        f"{os.path.splitext(os.path.basename(pdf_path))[0]}_highlighted.pdf"
    )

    with fitz.open(pdf_path) as doc:
        # Highlight sections in green
        for section, text in sections_to_highlight.items():
            if text:  # Ensure there is text to highlight
                cleaned_text = clean_text_overlay(text)  # Clean the text
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # Search for the entire cleaned section text to ensure full context is highlighted
                    highlight_instances = page.search_for(cleaned_text)

                    if highlight_instances:  # Check if highlights are found
                        for inst in highlight_instances:
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(stroke=(0, 1, 0), fill=(0, 1, 0))  # Green color
                            highlight.update()

        # Highlight titles in red
        for title in titles_to_highlight:
            title_cleaned = clean_text_overlay(title)  # Clean the title from special characters
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Search for the title text
                highlight_instances = page.search_for(title_cleaned)

                if highlight_instances:  # Check if highlights are found
                    for inst in highlight_instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(stroke=(1, 0, 0), fill=(1, 0, 0))  # Red color
                        highlight.update()

        # Save the modified PDF with both highlights
        doc.save(highlighted_file_path, garbage=4, deflate=True)

    return highlighted_file_path

def find_pdf_path(pdf_name: str) -> str:
  for root, _, files in os.walk(destination_base):
      if pdf_name in files:
          return os.path.join(root, pdf_name)
  return ""

@app.post("/pdf_overlay/")
def categorize_pdfs(request: PDFPaths):
    pdf1_path = request.pdf1_path
    pdf2_path = request.pdf2_path


    pdf1_path = find_pdf_path(pdf1_path)
    pdf2_path = find_pdf_path(pdf2_path)

    md_text_1 = convert_pdf_to_markdown_overlay(pdf1_path)
    md_text_2 = convert_pdf_to_markdown_overlay(pdf2_path)

    sections1 = {
    clean_title_overlay(sec): content
    for sec, content in extract_unique_sections_overlay(md_text_1).items()
    }
    sections2 = {
        clean_title_overlay(sec): content
        for sec, content in extract_unique_sections_overlay(md_text_2).items()
    }

    # Compare Sections for Similarities
    section_similarities_result = compare_sections_overlay(sections1, sections2)

    # Clean Section Similarities for Output
    cleaned_section_similarities = {
        clean_title_overlay(key): value for key, value in section_similarities_result.items()
    } if section_similarities_result else {}

    uncommon_titles = find_uncommon_sections_overlay(list(sections1.keys()), list(sections2.keys()))


    # Extract sections to highlight based on the similarity result (> threshold 0.1)
    sections_to_highlight_new = extract_sections_overlay(sections2, section_similarities_result, threshold=0.1)
    sections_to_highlight_old = extract_sections_overlay(sections1, section_similarities_result, threshold=0.1)


    # Combine titles into one list for highlighting
    all_titles = uncommon_titles['file1'] + uncommon_titles['file2']

    # Highlight sections and titles in the new PDF
    highlighted_pdf1_path = highlight_sections_and_titles_in_pdf_overlay(pdf1_path, sections_to_highlight_new, all_titles)

    # Highlight sections and titles in the old PDF
    highlighted_pdf2_path = highlight_sections_and_titles_in_pdf_overlay(pdf2_path, sections_to_highlight_old, all_titles)

    # Return the paths to the highlighted PDFs
    return {"highlighted_pdf1_path": highlighted_pdf1_path, "highlighted_pdf2_path": highlighted_pdf2_path}
