"""
RAG Document Analyzer - v3.14 (Complete) HF SPACES READY
Support for Mistral & OpenAI Embeddings
Author: Viral Patel
"""

import os
import json
import gc
import pickle
import traceback
import zipfile
import shutil
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import re
import uuid
import time
from functools import wraps
import threading

warnings.filterwarnings('ignore', message='urllib3.*doesn.*match')
warnings.filterwarnings('ignore', category=UserWarning, module='requests')

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pdfplumber
from tabulate import tabulate
import numpy as np
import requests
import faiss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Optional imports ---
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not installed — BM25 hybrid search disabled")

try:
    from sentence_transformers import CrossEncoder
    _RERANKER_MODEL = None
    def _get_reranker():
        global _RERANKER_MODEL
        if _RERANKER_MODEL is None:
            _RERANKER_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("✓ Cross-encoder reranker loaded")
        return _RERANKER_MODEL
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("sentence-transformers not installed — reranker disabled")

try:
    import camelot
    CAMELOT_AVAILABLE = True
    logger.info("✓ Camelot available")
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.info("camelot-py not installed — using pdfplumber only")

# Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.expanduser('~'), 'RAG_Uploads')
app.config['VECTORSTORE_FOLDER'] = os.path.join(os.path.expanduser('~'), 'RAG_Vectorstores')
app.config['FAISS_DB'] = os.path.join(os.path.expanduser('~'), 'RAG_FAISS_DB')
app.config['TEMP_FOLDER'] = os.path.join(os.path.expanduser('~'), 'RAG_TEMP')
app.config['MAX_PAGES_PER_PDF'] = 1000
app.config['MAX_TEXT_LEN_PER_PAGE'] = 500000
app.config['EMBEDDING_BATCH_SIZE'] = 16   # Optimised for stability
app.config['SESSION_TTL_HOURS'] = 24
app.config['MISTRAL_OCR_KEY'] = os.environ.get('MISTRAL_OCR_KEY', '')

# Safety: maximum characters per chunk (safe under 8192 tokens)
MAX_CHUNK_CHARS = 15000

for folder in [app.config['UPLOAD_FOLDER'], app.config['VECTORSTORE_FOLDER'],
               app.config['FAISS_DB'], app.config['TEMP_FOLDER']]:
    os.makedirs(folder, exist_ok=True)
    logger.info(f"✓ Ensured folder exists: {folder}")

ALLOWED_EXTENSIONS = {'pdf', 'zip'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== Retry & Rate Limiter ====================
def retry(max_retries=3, initial_delay=1, backoff=2, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {func.__name__}: {e}")
                    time.sleep(delay)
                    delay *= backoff
            return None
        return wrapper
    return decorator

_last_request_time = 0
_rate_limit_lock = threading.Lock()

def rate_limit(rps=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global _last_request_time
            with _rate_limit_lock:
                elapsed = time.time() - _last_request_time
                if elapsed < 1.0 / rps:
                    time.sleep(1.0 / rps - elapsed)
                _last_request_time = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ==================== PDF Processor ====================
class PDFProcessor:
    """Extract text from PDFs with page-level error handling and limits"""
    
    @staticmethod
    def extract_text_with_tables(pdf_path: str,
                                  mistral_ocr_key: str = "") -> List[Dict]:
        chunks = []
        filepath = pdf_path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        skipped_pages: Dict[int, str] = {}
        chunked_pages: List[int] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                filename = os.path.basename(pdf_path)
                total_pages = len(pdf.pages)

                if total_pages > app.config['MAX_PAGES_PER_PDF']:
                    logger.warning(
                        f"{filename}: {total_pages} pages exceeds limit, "
                        f"truncating to {app.config['MAX_PAGES_PER_PDF']}")
                    total_pages = app.config['MAX_PAGES_PER_PDF']

                logger.info(f"Extracting: {filename} ({total_pages} pages)")
                last_table_title = ""

                for page_num in range(1, total_pages + 1):
                    try:
                        page = pdf.pages[page_num - 1]
                        text = ""

                        # Step 1: Extract text (primary)
                        try:
                            text = page.extract_text() or ""
                        except Exception as e:
                            if "Unable to allocate output buffer" in str(e):
                                logger.error(f"{filename} p{page_num}: buffer allocation error — page skipped. ({e})")
                                skipped_pages[page_num] = "buffer"
                                continue
                            logger.warning(f"{filename} p{page_num}: extract_text() failed ({e}) — trying fallback")
                            text = ""

                        # Step 2: Fallback with relaxed tolerances
                        if not text.strip():
                            try:
                                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                                if text.strip():
                                    logger.info(f"{filename} p{page_num}: recovered via relaxed tolerances ({len(text)} chars)")
                            except:
                                pass

                        # Step 3: Second fallback via extract_words
                        if not text.strip():
                            try:
                                words = page.extract_words(x_tolerance=5, y_tolerance=5, keep_blank_chars=False)
                                if words:
                                    text = " ".join(w['text'] for w in words)
                                    if text.strip():
                                        logger.info(f"{filename} p{page_num}: recovered via extract_words ({len(words)} words)")
                            except:
                                pass

                        # Step 4: Truncate oversized pages
                        if len(text) > app.config['MAX_TEXT_LEN_PER_PAGE']:
                            logger.warning(f"{filename} p{page_num}: text {len(text)} chars, truncating to {app.config['MAX_TEXT_LEN_PER_PAGE']}")
                            text = text[:app.config['MAX_TEXT_LEN_PER_PAGE']]

                        # Step 5: Extract tables (3-tier)
                        mistral_key = app.config.get('MISTRAL_OCR_KEY', '')
                        try:
                            pdfplumber_tables = page.extract_tables(table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"})
                            if not pdfplumber_tables:
                                pdfplumber_tables = page.extract_tables(table_settings={"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"})
                            if not pdfplumber_tables:
                                pdfplumber_tables = page.extract_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})

                            raw_tables = pdfplumber_tables or []
                            tables_to_process: List[List[List[str]]] = []
                            used_camelot = False
                            used_ocr = False

                            for table in raw_tables:
                                cleaned = [[str(cell) if cell is not None else "" for cell in row] for row in table]
                                cleaned = PDFProcessor._collapse_merged_columns(cleaned)
                                quality = PDFProcessor._assess_table_quality(cleaned)
                                if quality >= 0.4:
                                    tables_to_process.append(cleaned)
                                else:
                                    if not used_camelot:
                                        camelot_tables = PDFProcessor._extract_tables_camelot(filepath, page_num)
                                        used_camelot = True
                                        if camelot_tables:
                                            for ct in camelot_tables:
                                                ct = PDFProcessor._collapse_merged_columns(ct)
                                                if PDFProcessor._assess_table_quality(ct) >= 0.4:
                                                    tables_to_process.append(ct)
                                                    logger.info(f"  p{page_num}: Camelot replaced low-quality pdfplumber table")

                            if not raw_tables:
                                if not used_camelot:
                                    camelot_tables = PDFProcessor._extract_tables_camelot(filepath, page_num)
                                    used_camelot = True
                                    tables_to_process.extend(camelot_tables)
                                if not tables_to_process and mistral_key:
                                    ocr_tables = PDFProcessor._extract_tables_mistral_ocr(filepath, page_num, mistral_key)
                                    used_ocr = True
                                    if ocr_tables:
                                        tables_to_process.extend(ocr_tables)
                                        logger.info(f"  p{page_num}: Mistral OCR extracted {len(ocr_tables)} table(s)")

                            for table_idx, cleaned_table in enumerate(tables_to_process):
                                try:
                                    if not cleaned_table:
                                        continue
                                    markdown_table = tabulate(cleaned_table, tablefmt="grid")
                                    text += f"\n\n{markdown_table}\n\n"

                                    detected_title = PDFProcessor._extract_table_title(text, table_idx)
                                    if detected_title:
                                        last_table_title = detected_title
                                    effective_text = text
                                    if not detected_title and last_table_title:
                                        effective_text = f"{last_table_title} (continued)\n" + text

                                    PDFProcessor._create_table_chunks(cleaned_table, page_num, page_num, table_idx, filename, chunks, page_text=effective_text)
                                except Exception as e:
                                    logger.warning(f"{filename} p{page_num} table {table_idx+1}: processing error ({e})")
                                    continue
                        except Exception as e:
                            logger.warning(f"{filename} p{page_num}: table extraction failed ({e}) — continuing with text only")

                        # Step 6: Semantic chunk the page text
                        if text.strip():
                            section_heading = PDFProcessor._extract_section_heading(text)
                            semantic_chunks = PDFProcessor._semantic_chunk_text(text, page_num, filename, section_heading)
                            chunks.extend(semantic_chunks)
                            chunked_pages.append(page_num)
                        else:
                            try:
                                has_images = len(page.images) > 0
                            except:
                                has_images = False
                            reason = "image_only" if has_images else "empty"
                            skipped_pages[page_num] = reason
                            logger.warning(f"{filename} p{page_num}: no text extracted ({'image/scanned page' if has_images else 'page appears blank'}) — page skipped")
                    except Exception as e:
                        logger.error(f"{filename} p{page_num}: unhandled error — page skipped. Reason: {str(e)}")
                        skipped_pages[page_num] = f"error: {str(e)[:80]}"
                        continue

                # Extraction report
                skipped_count = len(skipped_pages)
                skip_by_reason: Dict[str, List[int]] = {}
                for pn, reason in skipped_pages.items():
                    key = reason.split(":")[0]
                    skip_by_reason.setdefault(key, []).append(pn)

                if skipped_count == 0:
                    logger.info(f"✓ {filename}: all {total_pages} pages extracted — {len(chunks)} chunks created")
                else:
                    logger.warning(f"⚠ {filename}: {len(chunked_pages)}/{total_pages} pages chunked, {skipped_count} pages skipped")
                    for reason, pages in skip_by_reason.items():
                        sample = pages[:10]
                        more = len(pages) - 10
                        pg_str = ", ".join(str(p) for p in sample)
                        if more > 0:
                            pg_str += f" ... +{more} more"
                        logger.warning(f"   Skipped ({reason}): pages {pg_str}")
                    if "image_only" in skip_by_reason or "empty" in skip_by_reason:
                        logger.warning(f"   TIP: image_only/empty pages cannot be extracted by pdfplumber. If these pages contain critical data, run OCR on the PDF first (e.g. ocrmypdf).")
                return chunks
        except Exception as e:
            logger.error(f"PDF processing failed for {pdf_path}: {str(e)}")
            raise

    @staticmethod
    def _collapse_merged_columns(raw_table: List[List]) -> List[List[str]]:
        if not raw_table or len(raw_table) < 2:
            return [[str(c) if c else '' for c in row] for row in raw_table]

        header_row = raw_table[0]
        n_cols = len(header_row)

        header_positions = [i for i, h in enumerate(header_row) if h is not None and str(h).strip()]

        if not header_positions or len(header_positions) / n_cols >= 0.6:
            return [[str(c).replace('\n', ' ').strip() if c else '' for c in row] for row in raw_table]

        col_votes: Dict[int, int] = {}
        for row in raw_table[1:min(6, len(raw_table))]:
            for i, cell in enumerate(row):
                if cell is not None and str(cell).strip():
                    col_votes[i] = col_votes.get(i, 0) + 1

        n_logical = len(header_positions)
        populated = sorted([i for i, v in col_votes.items() if v >= 2])
        data_col_indices = populated[:n_logical] if len(populated) >= n_logical else populated

        if not data_col_indices:
            data_col_indices = header_positions

        header_labels = []
        for dc in data_col_indices:
            closest_hp = min(header_positions, key=lambda hp: abs(hp - dc))
            label = str(header_row[closest_hp]).strip()
            if dc == 0 and not label:
                label = 'Vessel Component'
            header_labels.append(label if label else f'Col{dc}')

        collapsed: List[List[str]] = [header_labels]
        for row in raw_table[1:]:
            new_row = [
                str(row[ci]).replace('\n', ' ').strip() if ci < len(row) and row[ci] else ''
                for ci in data_col_indices
            ]
            collapsed.append(new_row)
        return collapsed

    @staticmethod
    def _assess_table_quality(cleaned_table: List[List[str]]) -> float:
        if not cleaned_table or len(cleaned_table) < 2:
            return 0.0
        headers = [h for h in cleaned_table[0] if h.strip()]
        if len(headers) < 2:
            return 0.1
        total_cells = sum(len(row) for row in cleaned_table[1:])
        filled_cells = sum(1 for row in cleaned_table[1:] for cell in row if cell.strip())
        fill_ratio = filled_cells / max(total_cells, 1)
        col_score = min(len(headers) / 3, 1.0)
        return round((fill_ratio * 0.7 + col_score * 0.3), 3)

    @staticmethod
    def _extract_tables_camelot(pdf_path: str, page_num: int) -> List[List[List[str]]]:
        if not CAMELOT_AVAILABLE:
            return []
        tables_out = []
        for flavor in ('lattice', 'stream'):
            try:
                tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor=flavor, suppress_stdout=True)
                for t in tables:
                    if t.accuracy >= 60:
                        rows = [[str(cell).strip() for cell in row] for row in t.df.values.tolist()]
                        if len(rows) >= 2:
                            tables_out.append(rows)
                if tables_out:
                    logger.info(f"  Camelot ({flavor}) extracted {len(tables_out)} table(s) from page {page_num}")
                    return tables_out
            except Exception as e:
                logger.debug(f"  Camelot {flavor} failed page {page_num}: {e}")
                continue
        return []

    @staticmethod
    def _extract_tables_mistral_ocr(pdf_path: str, page_num: int, mistral_api_key: str) -> List[List[List[str]]]:
        if not mistral_api_key:
            return []
        try:
            import fitz, base64
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            doc.close()
            img_b64 = base64.b64encode(img_bytes).decode()
            response = requests.post(
                "https://api.mistral.ai/v1/ocr",
                headers={"Authorization": f"Bearer {mistral_api_key}", "Content-Type": "application/json"},
                json={"model": "mistral-ocr-latest", "document": {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}},
                timeout=30
            )
            response.raise_for_status()
            ocr_text = response.json().get("text", "")
            if not ocr_text:
                return []
            return PDFProcessor._parse_markdown_tables(ocr_text)
        except ImportError:
            logger.warning("PyMuPDF not installed for Mistral OCR fallback.")
            return []
        except Exception as e:
            logger.warning(f"  Mistral OCR failed page {page_num}: {e}")
            return []

    @staticmethod
    def _parse_markdown_tables(text: str) -> List[List[List[str]]]:
        import re
        tables = []
        lines = text.splitlines()
        current_table: List[List[str]] = []
        for line in lines:
            line = line.strip()
            if '|' in line and line.startswith('|'):
                if re.match(r'^\|[\s\-:|]+\|$', line.replace(' ', '')):
                    continue
                cells = [c.strip() for c in line.strip('|').split('|')]
                current_table.append(cells)
            else:
                if len(current_table) >= 2:
                    tables.append(current_table)
                current_table = []
        if len(current_table) >= 2:
            tables.append(current_table)
        return tables

    @staticmethod
    def _extract_table_title(page_text: str, table_idx: int) -> str:
        if not page_text:
            return ""
        table_pattern = re.compile(r'^(Table|TABLE|Fig\.|Figure|FIGURE|TABLE\s+UCS|Table\s+UCS|Table\s+UG|Table\s+UW)[\s\-][\w\-\.]+.*', re.IGNORECASE)
        lines = page_text.splitlines()
        captions = []
        for line in lines:
            line_stripped = line.strip()
            if table_pattern.match(line_stripped) and len(line_stripped) > 5:
                captions.append(line_stripped)
        if captions and table_idx < len(captions):
            return captions[table_idx]
        elif captions:
            return captions[-1]
        return ""

    @staticmethod
    def _extract_table_notes(page_text: str) -> str:
        if not page_text:
            return ""
        notes_pattern = re.compile(r'(GENERAL\s+NOTES?:|NOTES?:|General\s+Notes?:)', re.IGNORECASE)
        lines = page_text.splitlines()
        notes_lines = []
        in_notes = False
        for line in lines:
            if notes_pattern.match(line.strip()):
                in_notes = True
                notes_lines.append(line.strip())
                continue
            if in_notes:
                stripped = line.strip()
                if stripped and len(stripped) < 60 and stripped.isupper() and not stripped.startswith('('):
                    break
                notes_lines.append(stripped)
        if notes_lines:
            notes_text = " ".join(notes_lines)
            return notes_text[:800] if len(notes_text) > 800 else notes_text
        return ""

    @staticmethod
    def _create_table_chunks(cleaned_table: List[List[str]], page_num: int, original_page: int,
                            table_idx: int, filename: str, chunks: List[Dict],
                            page_text: str = "") -> None:
        try:
            if len(cleaned_table) < 2:
                return
            headers = [h.strip() for h in cleaned_table[0] if h and h.strip()]
            if not headers:
                return

            header_str = " | ".join(headers)
            table_title = PDFProcessor._extract_table_title(page_text, table_idx)
            title_prefix = f"{table_title}\n" if table_title else ""
            table_notes = PDFProcessor._extract_table_notes(page_text)
            notes_suffix = f"\n{table_notes}" if table_notes else ""

            try:
                full_table_md = tabulate(cleaned_table, headers="firstrow", tablefmt="grid")
                full_chunk_text = f"{title_prefix}Complete table (page {original_page}, table {table_idx + 1}):\n{full_table_md}{notes_suffix}"
                chunks.append({
                    'text': full_chunk_text,
                    'page_num': original_page,
                    'filename': filename,
                    'source': f"{filename} - Page {original_page} - Table {table_idx + 1} (full)",
                    'chunk_type': 'table-full'
                })
            except Exception as e:
                logger.warning(f"Full table chunk error on page {original_page}: {e}")

            for row_idx, row in enumerate(cleaned_table[1:], 1):
                try:
                    row_dict = {}
                    for i, header in enumerate(headers):
                        if i < len(row):
                            row_dict[header] = row[i].strip() if row[i] else ""
                        else:
                            row_dict[header] = ""
                    if not any(row_dict.values()):
                        continue
                    row_parts = [f"{k}: {v}" for k, v in row_dict.items() if v]
                    if not row_parts:
                        continue
                    row_text = " | ".join(row_parts)
                    row_chunk_text = f"{title_prefix}Table columns: {header_str}\nRow {row_idx}: {row_text}"
                    chunks.append({
                        'text': row_chunk_text,
                        'page_num': original_page,
                        'filename': filename,
                        'source': f"{filename} - Page {original_page} - Table {table_idx + 1}, Row {row_idx}",
                        'chunk_type': 'table-row'
                    })
                    if row_idx % 3 == 0:
                        combined_rows = []
                        start_idx = max(0, row_idx - 2)
                        for combine_idx in range(start_idx, min(row_idx + 1, len(cleaned_table) - 1)):
                            combine_row = cleaned_table[combine_idx + 1]
                            combine_dict = {}
                            for i, header in enumerate(headers):
                                if i < len(combine_row):
                                    combine_dict[header] = combine_row[i].strip() if combine_row[i] else ""
                                else:
                                    combine_dict[header] = ""
                            if any(combine_dict.values()):
                                combined_rows.append(" | ".join([f"{k}: {v}" for k, v in combine_dict.items() if v]))
                        if combined_rows:
                            combined_text = " ; ".join(combined_rows)
                            chunks.append({
                                'text': f"Table columns: {header_str}\nRows {max(1, start_idx)}-{row_idx}: {combined_text}",
                                'page_num': original_page,
                                'filename': filename,
                                'source': f"{filename} - Page {original_page} - Table {table_idx + 1}, Rows {max(1, start_idx)}-{row_idx}",
                                'chunk_type': 'table-row'
                            })
                except Exception as e:
                    logger.warning(f"Error processing table row {row_idx}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Error creating table chunks: {e}")

    @staticmethod
    def _semantic_chunk_text(text: str, page_num: int, filename: str,
                             section_heading: str, chunk_size: int = 400,
                             overlap: int = 80) -> List[Dict]:
        chunks = []
        words = text.split()
        if not words:
            return chunks
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            if section_heading:
                chunk_text = f"Section: {section_heading}\n{chunk_text}"
            chunks.append({
                'text': chunk_text,
                'page_num': page_num,
                'filename': filename,
                'source': f"{filename} - Page {page_num}",
                'chunk_type': 'text'
            })
            if end >= len(words):
                break
            start = end - overlap
        return chunks

    @staticmethod
    def _extract_section_heading(text: str) -> str:
        for line in text.splitlines():
            line = line.strip()
            if 3 <= len(line) <= 80 and (line.isupper() or (line[0].isupper() and len(line.split()) <= 8)):
                return line
        return ""

# ==================== Embedding Manager ====================
class EmbeddingManager:
    def __init__(self, provider: str, model: str, api_key: str):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        logger.info(f"✓ Using {self.provider} for embeddings (batch mode)")

    @retry(max_retries=4, initial_delay=2, backoff=2, exceptions=(requests.RequestException,))
    @rate_limit(rps=1)
    def get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            if self.provider == 'mistral':
                return self._get_mistral_embedding(text)
            elif self.provider == 'openai':
                return self._get_openai_embedding(text)
            return None
        except Exception as e:
            logger.error(f"Embedding error after retries: {str(e)}")
            return None

    @retry(max_retries=4, initial_delay=5, backoff=2)
    @rate_limit(rps=1)
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not texts:
            return []
        try:
            if self.provider == 'mistral':
                return self._get_mistral_embeddings_batch(texts)
            elif self.provider == 'openai':
                return self._get_openai_embeddings_batch(texts)
            return [None] * len(texts)
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return [None] * len(texts)

    def _get_mistral_embedding(self, text: str) -> Optional[List[float]]:
        response = requests.post(
            "https://api.mistral.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": [text]},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            return data['data'][0]['embedding']
        return None

    def _get_mistral_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        try:
            response = requests.post(
                "https://api.mistral.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "input": texts},
                timeout=120
            )
            if response.status_code != 200:
                logger.error(f"Mistral API {response.status_code}: {response.text[:500]}")
            response.raise_for_status()
            data = response.json()
            embeddings = [None] * len(texts)
            if 'data' in data:
                for item in data['data']:
                    idx = item['index']
                    if idx < len(embeddings):
                        embeddings[idx] = item['embedding']
            return embeddings
        except Exception as e:
            logger.error(f"Mistral batch embedding failed: {e}")
            raise

    def _get_openai_embedding(self, text: str) -> Optional[List[float]]:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": text},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            return data['data'][0]['embedding']
        return None

    def _get_openai_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": texts},
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        embeddings = [None] * len(texts)
        if 'data' in data:
            for item in data['data']:
                idx = item['index']
                if idx < len(embeddings):
                    embeddings[idx] = item['embedding']
        return embeddings

# ==================== Chat History Manager ====================
class ChatHistoryManager:
    MAX_TURNS = 10
    def __init__(self):
        self.history: List[Dict] = []
        self.last_answer: str = ""

    def add_user_message(self, message: str) -> None:
        self.history.append({"role": "user", "content": message, "timestamp": datetime.now().isoformat()})
        if len(self.history) > self.MAX_TURNS * 2:
            self.history = self.history[-(self.MAX_TURNS * 2):]

    def add_assistant_message(self, message: str) -> None:
        self.history.append({"role": "assistant", "content": message, "timestamp": datetime.now().isoformat()})
        self.last_answer = message
        if len(self.history) > self.MAX_TURNS * 2:
            self.history = self.history[-(self.MAX_TURNS * 2):]

    def get_messages_for_llm(self, n_turns: int = 6) -> List[Dict]:
        recent = self.history[-(n_turns * 2):]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def get_context_summary(self, n_turns: int = 3) -> str:
        if not self.history:
            return ""
        recent = self.history[-(n_turns * 2):]
        lines = ["CONVERSATION HISTORY (most recent first):"]
        for msg in reversed(recent):
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:400] + "..." if len(msg["content"]) > 400 else msg["content"]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def is_followup(self, query: str) -> bool:
        if not self.history:
            return False
        followup_signals = ['it ', 'its ', 'that ', 'those ', 'these ', 'they ',
            'the same', 'above', 'mentioned', 'you said', 'previous',
            'also', 'what about', 'and the', 'more about', 'elaborate',
            'explain more', 'tell me more', 'what else', 'any other',
            'furthermore', 'additionally', 'in that case', 'so then',
            'can you', 'could you', 'please', 'how about']
        q_lower = query.lower()
        return any(signal in q_lower for signal in followup_signals)

    def clear(self) -> None:
        self.history = []
        self.last_answer = ""

# ==================== FAISS Vector Store ====================
class FAISSVectorStore:
    DB_VERSION = 2
    def __init__(self, session_id: str, db_path: str):
        self.session_id = session_id
        self.db_path = db_path
        self.index = None
        self.documents = []
        self.metadata = []
        self.embedding_dim = 1024
        self.save_counter = 0
        self.SAVE_INTERVAL = 10
        self.bm25 = None
        self.bm25_corpus = []
        os.makedirs(db_path, exist_ok=True)
        self._load()

    def add_document(self, text: str, meta: Dict, embedding: List[float]) -> bool:
        try:
            if self.index is None:
                self.embedding_dim = len(embedding)
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            emb_array = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(emb_array)
            self.index.add(emb_array)
            self.documents.append(text)
            self.metadata.append(meta)
            self.bm25_corpus.append(text.lower().split())
            return True
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return False

    def _rebuild_bm25(self):
        if BM25_AVAILABLE and self.documents:
            self.bm25_corpus = [d.lower().split() for d in self.documents]
            self.bm25 = BM25Okapi(self.bm25_corpus)
            logger.info(f"✓ BM25 index built ({len(self.documents)} docs)")

    @property
    def _progress_path(self) -> str:
        return os.path.join(self.db_path, 'embed_progress.json')

    def save_embed_progress(self, completed_files: List[str], current_file: str, completed_chunks_in_file: int) -> None:
        state = {
            'completed_files': completed_files,
            'current_file': current_file,
            'completed_chunks_in_file': completed_chunks_in_file,
            'total_vectors_at_save': self.index.ntotal if self.index else 0,
            'saved_at': datetime.now().isoformat()
        }
        tmp = self._progress_path + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(state, f, indent=2)
            shutil.move(tmp, self._progress_path)
        except Exception as e:
            logger.error(f"Failed to save embed progress: {e}")

    def load_embed_progress(self) -> Dict:
        if not os.path.exists(self._progress_path):
            return {}
        try:
            with open(self._progress_path) as f:
                state = json.load(f)
            current_vectors = self.index.ntotal if self.index else 0
            saved_vectors = state.get('total_vectors_at_save', -1)
            if current_vectors != saved_vectors:
                logger.warning(f"Progress file mismatch: claims {saved_vectors} vectors but index has {current_vectors}. Discarding progress.")
                if current_vectors > saved_vectors > 0:
                    logger.info(f"Truncating index from {current_vectors} → {saved_vectors} vectors")
                    embs = np.zeros((saved_vectors, self.embedding_dim), dtype=np.float32)
                    for i in range(saved_vectors):
                        embs[i] = faiss.downcast_index(self.index).reconstruct(i)
                    new_index = faiss.IndexFlatIP(self.embedding_dim)
                    new_index.add(embs)
                    self.index = new_index
                    self.documents = self.documents[:saved_vectors]
                    self.metadata = self.metadata[:saved_vectors]
                    self.bm25_corpus = self.bm25_corpus[:saved_vectors]
                    return state
                return {}
            return state
        except Exception as e:
            logger.error(f"Failed to load embed progress: {e}")
            return {}

    def clear_embed_progress(self) -> None:
        try:
            if os.path.exists(self._progress_path):
                os.remove(self._progress_path)
                logger.info("✓ Embed progress file cleared")
        except Exception as e:
            logger.error(f"Failed to clear embed progress: {e}")

    def search(self, query_embedding: List[float], n_results: int = 12) -> List[Dict]:
        try:
            if self.index is None or self.index.ntotal == 0:
                return []
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            distances, indices = self.index.search(query_array, min(n_results, self.index.ntotal))
            faiss_results = {}
            for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
                if idx < len(self.documents) and float(score) >= 0.25:
                    faiss_results[int(idx)] = {
                        'text': self.documents[idx],
                        'source': self.metadata[idx].get('source', ''),
                        'page': str(self.metadata[idx].get('page', '')),
                        'filename': self.metadata[idx].get('filename', ''),
                        'chunk_type': self.metadata[idx].get('chunk_type', 'text'),
                        'distance': float(score),
                        'faiss_rank': rank
                    }
            results = sorted(faiss_results.values(), key=lambda x: x['distance'], reverse=True)
            return results[:n_results]
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

    def search_with_query(self, query_text: str, query_embedding: List[float], n_results: int = 12) -> List[Dict]:
        try:
            if self.index is None or self.index.ntotal == 0:
                return []
            candidate_n = min(n_results * 2, self.index.ntotal)
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            distances, indices = self.index.search(query_array, candidate_n)

            faiss_ranks = {}
            for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
                if idx < len(self.documents) and float(score) >= 0.25:
                    faiss_ranks[int(idx)] = {'rank': rank, 'score': float(score)}

            bm25_ranks = {}
            if BM25_AVAILABLE and self.bm25 is not None:
                try:
                    tokens = query_text.lower().split()
                    bm25_scores = self.bm25.get_scores(tokens)
                    bm25_top = np.argsort(bm25_scores)[::-1][:candidate_n]
                    for rank, idx in enumerate(bm25_top):
                        if bm25_scores[idx] > 0:
                            bm25_ranks[int(idx)] = {'rank': rank, 'score': float(bm25_scores[idx])}
                except Exception as e:
                    logger.warning(f"BM25 search error: {e}")

            K = 60
            all_ids = set(faiss_ranks.keys()) | set(bm25_ranks.keys())
            rrf_scores = {}
            for idx in all_ids:
                score = 0.0
                if idx in faiss_ranks:
                    score += 1.0 / (K + faiss_ranks[idx]['rank'] + 1)
                if idx in bm25_ranks:
                    score += 1.0 / (K + bm25_ranks[idx]['rank'] + 1)
                rrf_scores[idx] = score
            top_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:n_results]

            results = []
            for idx in top_ids:
                results.append({
                    'text': self.documents[idx],
                    'source': self.metadata[idx].get('source', ''),
                    'page': str(self.metadata[idx].get('page', '')),
                    'filename': self.metadata[idx].get('filename', ''),
                    'chunk_type': self.metadata[idx].get('chunk_type', 'text'),
                    'distance': faiss_ranks.get(idx, {}).get('score', 0.0),
                    'rrf_score': rrf_scores[idx]
                })
            return results
        except Exception as e:
            logger.error(f"Hybrid search error: {str(e)}")
            return self.search(query_embedding, n_results)

    def _save(self):
        try:
            if self.index is None:
                return
            temp_index = os.path.join(self.db_path, 'faiss.index.tmp')
            temp_data = os.path.join(self.db_path, 'data.pkl.tmp')
            faiss.write_index(self.index, temp_index)
            data = {'documents': self.documents, 'metadata': self.metadata, 'embedding_dim': self.embedding_dim}
            with open(temp_data, 'wb') as f:
                pickle.dump(data, f)
            shutil.move(temp_index, os.path.join(self.db_path, 'faiss.index'))
            shutil.move(temp_data, os.path.join(self.db_path, 'data.pkl'))
            meta = {'version': self.DB_VERSION, 'similarity': 'cosine', 'embedding_dim': self.embedding_dim, 'chunk_count': len(self.documents)}
            with open(os.path.join(self.db_path, 'meta.json'), 'w') as f:
                json.dump(meta, f)
            if BM25_AVAILABLE and self.bm25 is not None:
                bm25_path = os.path.join(self.db_path, 'bm25.pkl.tmp')
                with open(bm25_path, 'wb') as f:
                    pickle.dump({'bm25': self.bm25, 'corpus': self.bm25_corpus}, f)
                shutil.move(bm25_path, os.path.join(self.db_path, 'bm25.pkl'))
            logger.info(f"✓ Atomic save completed for {self.db_path}")
        except Exception as e:
            logger.error(f"Save error: {str(e)}")

    def _load(self):
        try:
            index_path = os.path.join(self.db_path, 'faiss.index')
            data_path = os.path.join(self.db_path, 'data.pkl')
            if not os.path.exists(index_path) or not os.path.exists(data_path):
                return False
            meta_path = os.path.join(self.db_path, 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get('version', 1) < self.DB_VERSION:
                    logger.warning("⚠ DB was built with v3.4 (L2). Cosine search will give wrong results. Please re-embed your PDFs.")
            else:
                logger.warning("⚠ No meta.json found — this DB may be from v3.4 (L2). Re-embedding recommended.")
            self.index = faiss.read_index(index_path)
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
            bm25_path = os.path.join(self.db_path, 'bm25.pkl')
            if os.path.exists(bm25_path) and BM25_AVAILABLE:
                with open(bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                self.bm25 = bm25_data.get('bm25')
                self.bm25_corpus = bm25_data.get('corpus', [])
                logger.info(f"✓ BM25 index loaded")
            else:
                self._rebuild_bm25()
            logger.info(f"✓ Loaded FAISS index with {self.index.ntotal} vectors")
            if self.index.ntotal != len(self.documents):
                logger.warning(f"⚠ Index integrity mismatch: {self.index.ntotal} vectors but {len(self.documents)} documents. Re-embed recommended.")
            return True
        except Exception as e:
            logger.error(f"Load error: {str(e)}")
            return False

    def load_from_zip(self, zip_path: str) -> bool:
        temp_dir = None
        try:
            if not zipfile.is_zipfile(zip_path):
                return False
            temp_id = str(uuid.uuid4())[:8]
            temp_dir = os.path.join(app.config['TEMP_FOLDER'], f'extract_{temp_id}')
            os.makedirs(temp_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)
            faiss_file = data_file = bm25_file = meta_file = None
            for root, dirs, files in os.walk(temp_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    if f == 'faiss.index':
                        faiss_file = fp
                    elif f == 'data.pkl':
                        data_file = fp
                    elif f == 'bm25.pkl':
                        bm25_file = fp
                    elif f == 'meta.json':
                        meta_file = fp
            if not faiss_file or not data_file:
                return False
            if meta_file:
                with open(meta_file) as f:
                    meta = json.load(f)
                if meta.get('version', 1) < self.DB_VERSION:
                    logger.warning("⚠ ZIP was built with v3.4 (L2 similarity). Re-embed recommended.")
                shutil.copy2(meta_file, os.path.join(self.db_path, 'meta.json'))
            else:
                logger.warning("⚠ ZIP has no meta.json — likely a v3.4 DB. Re-embed recommended.")
            shutil.copy2(faiss_file, os.path.join(self.db_path, 'faiss.index'))
            shutil.copy2(data_file, os.path.join(self.db_path, 'data.pkl'))
            if bm25_file:
                shutil.copy2(bm25_file, os.path.join(self.db_path, 'bm25.pkl'))
            success = self._load()
            if success and not bm25_file:
                self._rebuild_bm25()
            return success
        except Exception as e:
            logger.error(f"ZIP load error: {str(e)}")
            return False
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

# ==================== Session Management ====================
class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.uploaded_files = {}
        self.vector_store = FAISSVectorStore(session_id, os.path.join(app.config['FAISS_DB'], session_id))
        self.embedding_manager = None
        self.embedding_provider = None
        self.embedding_model = None
        self.chat_history = ChatHistoryManager()
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.progress = {"stage": "idle", "current": 0, "total": 0, "message": ""}
    def update_activity(self):
        self.last_activity = datetime.now()

sessions = {}

def cleanup_old_sessions():
    now = datetime.now()
    ttl = timedelta(hours=app.config['SESSION_TTL_HOURS'])
    to_delete = [sid for sid, sess in sessions.items() if now - sess.last_activity > ttl]
    for sid in to_delete:
        faiss_path = os.path.join(app.config['FAISS_DB'], sid)
        if os.path.exists(faiss_path):
            shutil.rmtree(faiss_path, ignore_errors=True)
        del sessions[sid]
        logger.info(f"Cleaned up session {sid}")
    if to_delete:
        logger.info(f"Cleaned up {len(to_delete)} old sessions")

# ==================== Flask Routes ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init-session', methods=['POST'])
def init_session():
    try:
        cleanup_old_sessions()
        session_id = os.urandom(16).hex()
        sessions[session_id] = Session(session_id)
        return jsonify({'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    try:
        session_id = request.form.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session'}), 400
        session = sessions[session_id]
        session.update_activity()
        uploaded = []
        errors = []
        for file in request.files.getlist('files'):
            try:
                if not allowed_file(file.filename):
                    errors.append(f"Invalid file: {file.filename}")
                    continue
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                session.uploaded_files[filename] = filepath
                uploaded.append({'filename': filename, 'size': os.path.getsize(filepath)})
            except Exception as e:
                errors.append(str(e))
        return jsonify({'success': len(uploaded) > 0, 'files': uploaded, 'errors': errors})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/progress/<session_id>', methods=['GET'])
def get_progress(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify(sessions[session_id].progress)

@app.route('/api/process', methods=['POST'])
def process_documents():
    session_id = None
    try:
        session_id = request.json.get('session_id')
        embedding_provider = request.json.get('embedding_provider')
        embedding_model = request.json.get('embedding_model')
        embedding_key = request.json.get('embedding_key')
        if not all([session_id, embedding_provider, embedding_model, embedding_key]):
            return jsonify({'error': 'Missing required parameters'}), 400
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404
        session = sessions[session_id]
        session.update_activity()
        if not session.uploaded_files:
            return jsonify({'error': 'No files uploaded'}), 400
        session.embedding_manager = EmbeddingManager(embedding_provider, embedding_model, embedding_key)
        session.embedding_provider = embedding_provider
        session.embedding_model = embedding_model

        batch_size = app.config['EMBEDDING_BATCH_SIZE']
        total_files = len(session.uploaded_files)

        resume = session.vector_store.load_embed_progress()
        completed_files = resume.get('completed_files', [])
        resume_file = resume.get('current_file', '')
        resume_chunks_done = resume.get('completed_chunks_in_file', 0)
        vectors_at_resume = resume.get('total_vectors_at_save', 0)

        if resume:
            logger.info(f"↩ Resuming embed: {len(completed_files)} files complete, partial file='{resume_file}' at chunk {resume_chunks_done}, vectors on disk={vectors_at_resume}")
            session.progress = {"stage": "resuming", "current": vectors_at_resume, "total": 0, "message": f"Resuming from interruption — {len(completed_files)} files already done..."}
        else:
            if session.vector_store.index is not None and session.vector_store.index.ntotal > 0:
                logger.warning("No progress file found but index has vectors — clearing stale index for clean start")
                session.vector_store.index = None
                session.vector_store.documents = []
                session.vector_store.metadata = []
                session.vector_store.bm25 = None
                session.vector_store.bm25_corpus = []
            session.progress = {"stage": "extracting", "current": 0, "total": total_files, "message": "Extracting text from PDFs..."}

        successful_embeds = 0
        total_chunks = 0
        skipped_chunks = 0

        for file_idx, (filename, filepath) in enumerate(session.uploaded_files.items(), 1):
            if filename in completed_files:
                logger.info(f"↩ Skipping '{filename}' — already fully embedded")
                session.progress["message"] = f"Skipping {filename} (already done) — {file_idx}/{total_files}"
                continue

            session.progress["current"] = file_idx
            session.progress["message"] = f"Processing {filename} ({file_idx}/{total_files})..."

            try:
                chunks = PDFProcessor.extract_text_with_tables(filepath, mistral_ocr_key=app.config.get('MISTRAL_OCR_KEY', ''))
            except Exception as e:
                logger.error(f"Failed to extract from {filename}: {e}")
                session.progress["message"] = f"Extraction error on {filename}: {str(e)}"
                continue

            total_chunks_in_file = len(chunks)
            start_chunk = 0
            if filename == resume_file and resume_chunks_done > 0:
                start_chunk = resume_chunks_done
                logger.info(f"↩ Resuming '{filename}' from chunk {start_chunk}/{total_chunks_in_file}")
                session.progress["message"] = f"Resuming '{filename}' from chunk {start_chunk}/{total_chunks_in_file}"

            chunks_done_this_file = start_chunk

            for batch_start in range(start_chunk, total_chunks_in_file, batch_size):
                batch_end = min(batch_start + batch_size, total_chunks_in_file)
                batch_chunks = chunks[batch_start:batch_end]
                batch_texts = [c['text'] for c in batch_chunks]

                # Truncate oversize chunks to safe length
                truncated_any = False
                for i, t in enumerate(batch_texts):
                    if len(t) > MAX_CHUNK_CHARS:
                        logger.warning(f"Truncating chunk from {len(t)} to {MAX_CHUNK_CHARS} chars: {batch_chunks[i].get('source','unknown')}")
                        batch_texts[i] = t[:MAX_CHUNK_CHARS]
                        truncated_any = True

                batch_embeddings = session.embedding_manager.get_embeddings_batch(batch_texts)

                batch_added = 0
                for chunk, embedding in zip(batch_chunks, batch_embeddings):
                    if embedding is None:
                        logger.warning(f"Skipping chunk (embed failed after retries): {chunk.get('source','unknown')}")
                        skipped_chunks += 1
                        continue
                    metadata = {'source': chunk['source'], 'page': chunk['page_num'], 'filename': chunk['filename'], 'chunk_type': chunk.get('chunk_type', 'text')}
                    session.vector_store.add_document(chunk['text'], metadata, embedding)
                    successful_embeds += 1
                    batch_added += 1
                    total_chunks += 1

                chunks_done_this_file += len(batch_chunks)
                session.vector_store._save()
                session.vector_store.save_embed_progress(completed_files, filename, chunks_done_this_file)

                # Adaptive sleep – only if we didn't truncate (original chunks were within limit)
                if not truncated_any:
                    batch_tokens = sum(len(t.split()) * 1.3 for t in batch_texts)
                    sleep_needed = (batch_tokens / 10_000_000) * 60
                    if sleep_needed > 0.1:
                        time.sleep(sleep_needed)

                session.progress["stage"] = "embedding"
                session.progress["current"] = session.vector_store.index.ntotal if session.vector_store.index else total_chunks
                session.progress["message"] = f"{filename}: batch {batch_start//batch_size + 1} done ({chunks_done_this_file}/{total_chunks_in_file} chunks) — {successful_embeds} embedded total"
                gc.collect()

            completed_files.append(filename)
            session.vector_store.save_embed_progress(completed_files, '', 0)
            logger.info(f"✓ '{filename}' fully embedded ({chunks_done_this_file} chunks processed)")
            del chunks
            gc.collect()

        session.vector_store._save()
        if successful_embeds == 0 and not resume:
            session.progress = {"stage": "error", "message": "No embeddings generated. Check your API key and model name."}
            return jsonify({'error': 'No embeddings generated'}), 500

        session.vector_store.clear_embed_progress()
        total_vectors = session.vector_store.index.ntotal if session.vector_store.index else 0
        session.progress = {"stage": "completed", "current": total_vectors, "total": total_vectors, "message": f"✓ All files processed — {total_vectors} vectors in index ({successful_embeds} new embeddings this run)"}
        return jsonify({'success': True, 'pages_processed': total_chunks, 'chunks_created': total_chunks, 'successful_embeds': successful_embeds, 'skipped_chunks': skipped_chunks, 'total_vectors': total_vectors, 'files_skipped': len([f for f in completed_files if f in session.uploaded_files and f not in [fn for fn,_ in list(session.uploaded_files.items())[-total_chunks:] if total_chunks]])})
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        logger.error(traceback.format_exc())
        if session_id and session_id in sessions:
            sessions[session_id].progress = {"stage": "error", "message": f"Error: {str(e)} — restart and click Process to resume"}
        return jsonify({'error': str(e)}), 500

# ==================== Helper Functions for Query ====================
@retry(max_retries=2, initial_delay=1)
def _expand_query(query: str, chat_history_summary: str,
                  provider: str, model: str, api_key: str) -> str:
    prompt = f"""Expand this engineering query for document retrieval.

RULES:
1. Expand ALL abbreviations (RF→Reinforcing Pad, SRN→Self-Reinforced Nozzle, PWHT→Post Weld Heat Treatment)
2. If query asks when X is NOT allowed, include BOTH:
   - Prohibition terms: "X not permitted", "X prohibited"
   - Alternative requirement: "Y shall be used", "Y required"
3. Add standard terminology variations (spec vs field terms)
4. Keep output under 3 sentences

CONTEXT: {chat_history_summary if chat_history_summary else 'None — this is the first question.'}
QUERY: {query}

Output only the expanded query:"""

    messages = [{"role": "user", "content": prompt}]
    try:
        url = ("https://api.deepseek.com/chat/completions"
               if provider.lower() == 'deepseek'
               else "https://api.openai.com/v1/chat/completions")

        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model, "messages": messages,
                  "temperature": 0.1, "max_tokens": 250},
            timeout=15
        )
        response.raise_for_status()
        expanded = response.json()['choices'][0]['message']['content'].strip()
        if expanded and len(expanded) > 10:
            logger.info(f"Query expanded: '{query[:60]}' → '{expanded[:100]}'")
            return expanded
    except Exception as e:
        logger.warning(f"Query expansion failed ({e}) — using original query")
    return query

@retry(max_retries=1, initial_delay=1)
def _hyde_generate(query: str, provider: str, model: str, api_key: str) -> str:
    prompt = f"""You are writing a clause for an oil & gas engineering specification.
Generate a hypothetical clause that would answer this question.
Use realistic values typical of industry standards (ASME, API, IEC, IEEE, ISA, NACE, ASTM, BS, EN)and company specifications from major operators.
Format: Formal spec language with specific values/thresholds.
Max 4 sentences. Do NOT include preamble or explanation.

Question: {query}

Hypothetical clause:"""

    try:
        url = ("https://api.deepseek.com/chat/completions"
               if provider.lower() == 'deepseek'
               else "https://api.openai.com/v1/chat/completions")
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1, "max_tokens": 200},
            timeout=15
        )
        response.raise_for_status()
        hyde_text = response.json()['choices'][0]['message']['content'].strip()
        if hyde_text and len(hyde_text) > 20:
            logger.info(f"HyDE generated ({len(hyde_text)} chars)")
            return hyde_text
    except Exception as e:
        logger.warning(f"HyDE generation failed ({e}) — using original embedding only")
    return ""

@retry(max_retries=1, initial_delay=1)
def _multi_query_generate(query: str, provider: str,
                           model: str, api_key: str) -> List[str]:
    prompt = f"""Generate exactly 2 alternative phrasings of this engineering query:

1. Regulatory/standards perspective (using ASME, API, IEC, IEEE, ISA, NACE, ASTM, BS, EN code references and formal terminology)
2. Practical implementation perspective (using field/contractor terminology)

Output EXACTLY 2 lines — one phrasing per line — no numbering, no labels, no explanation.

ORIGINAL QUERY: {query}"""

    try:
        url = ("https://api.deepseek.com/chat/completions"
               if provider.lower() == 'deepseek'
               else "https://api.openai.com/v1/chat/completions")
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3, "max_tokens": 200},
            timeout=15
        )
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content'].strip()
        variants = [line.strip() for line in content.splitlines()
                    if line.strip() and len(line.strip()) > 10][:2]
        if variants:
            logger.info(f"Multi-query: {len(variants)} variants generated")
            return variants
    except Exception as e:
        logger.warning(f"Multi-query generation failed ({e}) — single query only")
    return []

def _is_section_boundary(chunk_text: str) -> bool:
    """Detect if chunk is a section header or boundary marker."""
    text = chunk_text.strip()
    # Common section patterns in engineering specs
    patterns = [
        r'^\d+\.\d+\s+[A-Z]',  # "3.1 MATERIAL REQUIREMENTS"
        r'^[A-Z\s]{10,}$',      # "GENERAL REQUIREMENTS"
        r'^\d+\s+[A-Z]',        # "5 TESTING AND INSPECTION"
        r'^APPENDIX\s+[A-Z0-9]', # "APPENDIX A"
        r'^TABLE\s+\d+',        # "TABLE 3"
        r'^FIGURE\s+\d+',       # "FIGURE 2"
    ]
    return any(re.match(p, text) for p in patterns)

def _get_sentence_window(chunk_text: str, source: str,
                          all_chunks: List[Dict], window: int = 1) -> str:
    """Get context window with boundary awareness and token limit."""
    try:
        same_source = [c for c in all_chunks
                       if c.get('source', '') == source
                       and c.get('chunk_type', 'text') != 'table-full']

        if len(same_source) <= 1:
            return chunk_text

        chunk_idx = None
        for i, c in enumerate(same_source):
            if c['text'] == chunk_text:
                chunk_idx = i
                break

        if chunk_idx is None:
            return chunk_text

        # Expand window but respect boundaries
        start = chunk_idx
        end = chunk_idx + 1
        
        # Expand backward
        for i in range(chunk_idx - 1, max(0, chunk_idx - window) - 1, -1):
            if _is_section_boundary(same_source[i]['text']):
                break
            if same_source[i].get('chunk_type') != same_source[chunk_idx].get('chunk_type'):
                break
            start = i
        
        # Expand forward
        for i in range(chunk_idx + 1, min(len(same_source), chunk_idx + window + 1)):
            if _is_section_boundary(same_source[i]['text']):
                break
            if same_source[i].get('chunk_type') != same_source[chunk_idx].get('chunk_type'):
                break
            end = i + 1

        window_texts = [c['text'] for c in same_source[start:end]]
        combined = " ".join(window_texts)

        # Limit to ~500 tokens (rough estimate: 1 token ≈ 4 chars)
        if len(combined) > 2000:
            combined = combined[:2000] + "..."

        # Deduplicate sentences
        sentences = combined.split('. ')
        seen = set()
        deduped = []
        for s in sentences:
            key = s.strip()[:60]
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        return '. '.join(deduped)

    except Exception:
        return chunk_text

def _cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    a = np.array(emb1, dtype=np.float32)
    b = np.array(emb2, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def _deduplicate_results(results: List[Dict], threshold: float = 0.85) -> List[Dict]:
    """Remove near-duplicate chunks using semantic similarity."""
    if not results:
        return results
    
    kept = []
    for r in results:
        is_duplicate = False
        r_embedding = r.get('embedding')
        
        if r_embedding is None:
            # If no embedding, fall back to text comparison
            for k in kept:
                if r.get('source') == k.get('source'):
                    # Simple text overlap check
                    r_text = r['text'][:100].lower()
                    k_text = k['text'][:100].lower()
                    overlap = len(set(r_text.split()) & set(k_text.split()))
                    if overlap > 0.7 * min(len(r_text.split()), len(k_text.split())):
                        is_duplicate = True
                        break
        else:
            for k in kept:
                k_embedding = k.get('embedding')
                if k_embedding is not None:
                    sim = _cosine_similarity(r_embedding, k_embedding)
                    if sim > threshold:
                        is_duplicate = True
                        logger.debug(f"Duplicate found (similarity: {sim:.3f})")
                        break
        
        if not is_duplicate:
            kept.append(r)
    
    logger.info(f"Deduplication: {len(results)} → {len(kept)} unique chunks")
    return kept

def _detect_conflicts(results: List[Dict]) -> str:
    """Detect conflicting values/requirements across sources."""
    conflicts = []
    
    # Extract numeric values with units
    value_pattern = r'(\d+(?:\.\d+)?)\s*(°C|°F|mm|MPa|psi|bar|inches?|hours?|days?)'
    
    value_map = {}  # {parameter: [(value, unit, source), ...]}
    
    for r in results:
        text = r['text']
        source = r.get('source', 'Unknown')
        
        # Find all numeric values
        matches = re.finditer(value_pattern, text, re.IGNORECASE)
        for match in matches:
            value = float(match.group(1))
            unit = match.group(2).lower()
            
            # Try to extract parameter name (crude heuristic)
            start = max(0, match.start() - 50)
            context = text[start:match.start()]
            param_words = re.findall(r'\b[A-Z][a-z]+\b', context)
            param = ' '.join(param_words[-3:]) if param_words else 'value'
            
            key = f"{param}_{unit}"
            if key not in value_map:
                value_map[key] = []
            value_map[key].append((value, unit, source))
    
    # Check for conflicts
    for param, values in value_map.items():
        if len(values) > 1:
            unique_values = set(v[0] for v in values)
            if len(unique_values) > 1:
                conflict_str = f"CONFLICT in {param.replace('_', ' ')}: "
                conflict_str += ", ".join([f"{v[0]} {v[1]} ({v[2]})" for v in values])
                conflicts.append(conflict_str)
    
    if conflicts:
        logger.warning(f"Detected {len(conflicts)} potential conflicts")
        return "\n".join(conflicts)
    return ""

def _classify_query_complexity(query: str) -> str:
    """Classify query as simple_factual or complex."""
    simple_patterns = [
        r'^what is (the )?',
        r'^define ',
        r'^meaning of ',
        r'maximum (temperature|pressure|thickness)',
        r'minimum (temperature|pressure|thickness)',
        r'^list ',
        r'^name ',
    ]
    
    query_lower = query.lower().strip()
    
    for pattern in simple_patterns:
        if re.match(pattern, query_lower):
            return 'simple_factual'
    
    return 'complex'

@app.route('/api/query', methods=['POST'])
def query_documents():
    try:
        session_id = request.json.get('session_id')
        query = request.json.get('query')
        llm_provider = request.json.get('llm_provider')
        llm_model = request.json.get('llm_model')
        llm_key = request.json.get('llm_key')
        embedding_provider = request.json.get('embedding_provider')
        embedding_model = request.json.get('embedding_model')
        embedding_key = request.json.get('embedding_key')

        if not all([session_id, query, llm_provider, llm_model, llm_key]):
            return jsonify({'error': 'Missing required parameters'}), 400
        if session_id not in sessions:
            return jsonify({'error': 'Session not found'}), 404

        session = sessions[session_id]
        session.update_activity()

        if session.vector_store.index is None:
            return jsonify({'error': 'No documents processed yet'}), 400

        if not session.embedding_manager:
            if not embedding_key:
                return jsonify({'error': 'Embedding API Key required to search vectorstore'}), 400
            session.embedding_manager = EmbeddingManager(embedding_provider or 'mistral',
                                                          embedding_model or 'mistral-embed',
                                                          embedding_key)

        # Classify query complexity
        query_complexity = _classify_query_complexity(query)
        logger.info(f"Query complexity: {query_complexity}")

        # Step 1: generate all text variants first (LLM calls only, no embeddings yet)
        chat_summary = session.chat_history.get_context_summary(n_turns=3)
        expanded_query = _expand_query(
            query, chat_summary, llm_provider, llm_model, llm_key)
        
        # Only use HyDE and multi-query for complex queries
        if query_complexity == 'complex':
            hyde_text = _hyde_generate(expanded_query, llm_provider, llm_model, llm_key)
            variants = _multi_query_generate(
                expanded_query, llm_provider, llm_model, llm_key)
        else:
            hyde_text = ""
            variants = []
            logger.info("Skipping HyDE and multi-query for simple factual query")

        # Step 2: collect all texts that need embedding into one list
        embed_texts = [query]
        if expanded_query != query:
            embed_texts.append(expanded_query)
        if hyde_text:
            embed_texts.append(hyde_text)
        embed_texts.extend(variants)

        # Step 3: single batch call — 1 API request instead of 5-6
        all_embeddings = session.embedding_manager.get_embeddings_batch(embed_texts)
        embed_map = {text: emb for text, emb in zip(embed_texts, all_embeddings) if emb is not None}

        query_embedding = embed_map.get(query)
        if not query_embedding:
            return jsonify({'error': 'Failed to embed query'}), 500

        # Use expanded query embedding if available (better than raw query)
        if expanded_query != query and expanded_query in embed_map:
            query_embedding = embed_map[expanded_query]

        # Merge HyDE embedding with query embedding (0.7 query / 0.3 HyDE weighting)
        if hyde_text and hyde_text in embed_map:
            q_arr = np.array(query_embedding, dtype=np.float32)
            h_arr = np.array(embed_map[hyde_text], dtype=np.float32)
            combined = 0.7 * q_arr + 0.3 * h_arr
            norm = np.linalg.norm(combined)
            if norm > 0:
                query_embedding = (combined / norm).tolist()
            logger.info("HyDE embedding merged with query (0.7/0.3 weight)")

        # Search with each variant using its embedding from the batch
        variant_results_list = []
        for variant in variants:
            if variant in embed_map:
                v_results = session.vector_store.search_with_query(
                    variant, embed_map[variant], n_results=8)
                variant_results_list.append(v_results)

        primary_results = session.vector_store.search_with_query(
            expanded_query, query_embedding, n_results=20)

        if variant_results_list:
            K = 60
            rrf_scores: Dict[str, float] = {}
            rrf_data: Dict[str, Dict] = {}

            def add_to_rrf(results, weight=1.0):
                for rank, r in enumerate(results):
                    key = r.get('source', '') + r['text'][:50]
                    rrf_scores[key] = rrf_scores.get(key, 0) + weight / (K + rank + 1)
                    rrf_data[key] = r

            # Higher weight for primary (expanded query uses domain knowledge)
            add_to_rrf(primary_results, weight=3.0)
            for vr in variant_results_list:
                add_to_rrf(vr, weight=0.5)

            merged = sorted(rrf_data.values(),
                            key=lambda r: rrf_scores.get(
                                r.get('source','') + r['text'][:50], 0),
                            reverse=True)[:15]
            search_results = merged
            logger.info(f"Multi-query merge: {len(search_results)} candidates "
                        f"from {1 + len(variant_results_list)} query variants")
        else:
            search_results = primary_results

        # Deduplicate before reranking
        search_results = _deduplicate_results(search_results, threshold=0.85)

        if RERANKER_AVAILABLE and search_results:
            try:
                reranker = _get_reranker()
                pairs = [(query, r['text']) for r in search_results]
                scores = reranker.predict(pairs)
                for r, s in zip(search_results, scores):
                    r['rerank_score'] = float(s)
                search_results = sorted(
                    search_results,
                    key=lambda x: x.get('rerank_score', 0),
                    reverse=True)[:10]  # Increased from 8 to 10
                logger.info(f"✓ Reranker applied, top-10 from {len(pairs)} candidates")
            except Exception as e:
                logger.warning(f"Reranker failed: {e}")
                search_results = search_results[:10]
        else:
            search_results = search_results[:10]

        if not search_results:
            response_text = "I could not find relevant documents matching your query. Please try a different question."
            sources = []
            conflict_warning = ""
        else:
            all_chunks = session.vector_store.documents
            all_meta = session.vector_store.metadata

            chunk_dicts = [
                {'text': t, 'source': m.get('source',''),
                 'chunk_type': m.get('chunk_type','text')}
                for t, m in zip(all_chunks, all_meta)
            ]

            # Detect conflicts
            conflict_warning = _detect_conflicts(search_results)

            context_parts = ["RELEVANT DOCUMENTS:\n" + "=" * 50]
            
            if conflict_warning:
                context_parts.append(f"\n⚠️ DETECTED CONFLICTS:\n{conflict_warning}\n")
            
            for result in search_results:
                if result.get('chunk_type', 'text') == 'text':
                    expanded_text = _get_sentence_window(
                        result['text'], result['source'], chunk_dicts, window=1)
                else:
                    expanded_text = result['text']

                context_parts.append(
                    f"\nSource: {result['source']}\nContent: {expanded_text}\n")
            context = "\n".join(context_parts)

            chat_messages = session.chat_history.get_messages_for_llm(n_turns=3)
            is_followup = session.chat_history.is_followup(query)

            response_text = _query_llm(
                llm_provider, llm_model, llm_key,
                query, context, chat_messages,
                is_followup=is_followup,
                expanded_query=expanded_query if expanded_query != query else None,
                has_conflicts=bool(conflict_warning)
            )
            if not response_text:
                response_text = "I couldn't generate a response. Please try again."

            sources = list({
                f"{result['filename']} — Page {result['page']}"
                for result in search_results
            })

        session.chat_history.add_user_message(query)
        session.chat_history.add_assistant_message(response_text)

        return jsonify({
            'success': True,
            'response': response_text,
            'sources': sources,
            'expanded_query': expanded_query if expanded_query != query else None
        })

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@retry(max_retries=2, initial_delay=1)
def _query_llm(provider: str, model: str, api_key: str,
               question: str, context: str,
               chat_messages: List[Dict] = None,
               is_followup: bool = False,
               expanded_query: str = None,
               has_conflicts: bool = False) -> str:

    followup_note = (
        "\nNOTE: This question is a follow-up to the previous answer. "
        "You may reference your previous answer when relevant, "
        "but always ground new facts in the RELEVANT DOCUMENTS section."
        if is_followup else ""
    )

    expanded_note = (
        f"\nQUERY EXPANSION: The user asked '{question}'. "
        f"This was interpreted as: '{expanded_query}' for retrieval purposes."
        if expanded_query else ""
    )

    conflict_note = (
        "\n⚠️ CONFLICTS DETECTED: The retrieved documents contain conflicting values. "
        "You MUST explicitly flag these conflicts in your answer."
        if has_conflicts else ""
    )

    system_prompt = f"""You are a precise technical document assistant for a company's internal engineering knowledge system.
You are having an ongoing conversation — you remember everything said earlier in this chat.{followup_note}{conflict_note}

DOMAIN REASONING RULES:
1. Before answering, identify ALL abbreviations in the question and expand them.
   Common examples: RF pad = Reinforcing Pad, IRN = Integrally Reinforced Nozzle,
   SRN = Self-Reinforced Nozzle, PWHT = Post Weld Heat Treatment,
   NDE = Non-Destructive Examination.

2. PROHIBITION-IMPLICATION RULE — CRITICAL:
   When the document says something is NOT ALLOWED, PROHIBITED, or SHALL NOT BE USED,
   you must ALWAYS state what SHALL be used instead, using your engineering knowledge
   to make the connection even if the alternative is in a different section.
   Examples:
   - "RF pad / Reinforcing pad not allowed" → state that integrally reinforced nozzle
     (SRN — Self-Reinforced Nozzle / set-through reinforcement) must be used instead
   - "Socket weld not permitted" → state that butt weld is required
   - "Seamless pipe required" → state that welded pipe is not acceptable
   This connection MUST appear in your answer even if the retrieved documents only
   contain the prohibition and not the alternative requirement.

3. Do NOT work like a keyword search. Apply engineering judgment to interpret what the
   question is really asking, then find the answer in the documents.{expanded_note}

4. When a prohibition applies to specific service conditions (lethal, sour, cyclic,
   high temperature, hydrogen), list EACH condition separately with its threshold value.

5. COMPLETENESS RULE — CRITICAL: If the question asks about a general requirement
   (e.g. "when to perform PWHT", "what are the radiography requirements",
   "what materials are permitted") and the documents contain information for MULTIPLE
   material groups, P-Numbers, thickness ranges, or categories — you MUST include
   ALL of them in your answer, not just the first one you find.
   Never answer for only one P-Number or group when the question is general.
   Organise by group: "For P-No. 1: ... For P-No. 3: ... For P-No. 4: ..."

ANSWER RULES:
1. Answer ONLY from the RELEVANT DOCUMENTS section. Do not use outside knowledge for facts.
2. Quote exact values, dimensions, and specifications as they appear in the source.
3. If a table contains the answer, reproduce the relevant rows in clean formatted text.
4. If documents contain only partial information, state what you found AND what is missing.
5. If the same value appears in multiple sources with conflicting data, flag the conflict explicitly.
6. COMPLETENESS RULE — CRITICAL: You MUST capture and include ALL relevant requirements
   from ALL provided documents — base specifications, addenda, amendments, appendices,
   and supplements. Never omit a requirement because it appears in a secondary document.
   If an addendum or amendment modifies a base specification clause, include BOTH the
   original requirement AND the amendment, clearly stating which document each comes from.
7. If the question asks about a general requirement and the documents contain information
   for MULTIPLE material groups, P-Numbers, thickness ranges, or categories — include
   ALL of them in your answer. Never answer for only one group when the question is general.
   Organise by group: "For P-No. 1: ... For P-No. 3: ... For P-No. 4: ..."
8. If no relevant information is found, respond exactly:
   "The provided documents do not contain sufficient information to answer this question."

OUTPUT FORMAT — follow this structure:

**Answer**

[Write your answer here following these rules:
 - Use bullet points (•) for listing multiple items, requirements, or specifications
 - Use **bold** to highlight key terms, values, material grades, temperatures, thicknesses
 - Use plain paragraphs when explaining a concept or reasoning
 - When a bullet point has a category label (e.g. High Temperature, Sour Service),
   format it as: • **Category Label:** explanation text
 - If the answer involves table data, present each row as a bullet point
 - Each bullet point must be a complete standalone statement
 - Bold only the most important technical terms — do not bold every word]

**Conclusion**

[Write a concise summary conclusion in a separate paragraph.
 The word CONCLUSION must appear on its own line above the conclusion text.
 Use **bold** for key values or terms in the conclusion if needed.]

**Confidence: X/5**

[X = 5: Answer found directly and explicitly in the documents
 X = 4: Answer clearly implied by document content
 X = 3: Partial answer found — some aspects not covered
 X = 2: Answer inferred — not stated directly, use with caution
 X = 1: Very limited relevant content found — verify manually
One sentence explaining the confidence score.]

**Sources**

[List each source on a new line as:
 - Filename — Page N: one sentence describing what this source contributed]"""

    messages = [{"role": "system", "content": system_prompt}]

    if chat_messages:
        messages.extend(chat_messages)

    messages.append({
        "role": "user",
        "content": f"{context}\n\nQuestion: {question}"
    })

    if provider.lower() == 'deepseek':
        url = "https://api.deepseek.com/chat/completions"
    else:
        url = "https://api.openai.com/v1/chat/completions"

    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2500
        },
        timeout=60
    )
    response.raise_for_status()
    data = response.json()

    if 'choices' in data and len(data['choices']) > 0:
        response_text = data['choices'][0]['message']['content']

        # Keep ** bold and • bullets — only remove ## headers and clean whitespace
        response_text = re.sub(r'^#+\s*', '', response_text, flags=re.MULTILINE)

        lines = response_text.split('\n')
        cleaned = []
        prev_empty = False
        for line in lines:
            line = line.rstrip()
            if not line:
                if not prev_empty:
                    cleaned.append('')
                    prev_empty = True
            else:
                cleaned.append(line)
                prev_empty = False
        return '\n'.join(cleaned).strip()

    return ""

# ==================== Additional Routes ====================
@app.route('/api/download-vectorstore', methods=['POST'])
def download_vectorstore():
    try:
        session_id = request.json.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session'}), 400
        session = sessions[session_id]
        session_db_path = os.path.join(app.config['FAISS_DB'], session_id)
        if not os.path.exists(session_db_path):
            return jsonify({'error': 'No vectorstore data'}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f'vectorstore_{session_id[:8]}_{timestamp}.zip'
        zip_path = os.path.join(app.config['VECTORSTORE_FOLDER'], zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(session_db_path):
                for file in files:
                    if file.endswith('.tmp'):
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, session_db_path)
                    zipf.write(file_path, arcname)
        return send_file(zip_path, as_attachment=True, download_name=zip_filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify-index', methods=['POST'])
def verify_index():
    try:
        session_id = request.json.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session'}), 400
        session = sessions[session_id]
        vs = session.vector_store
        total_vectors = vs.index.ntotal if vs.index else 0
        total_documents = len(vs.documents)
        match = total_vectors == total_documents
        return jsonify({
            'total_vectors': total_vectors,
            'total_documents': total_documents,
            'index_healthy': match,
            'warning': None if match else f"Mismatch: {total_vectors} vectors vs {total_documents} documents — re-embed recommended"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-vectorstore', methods=['POST'])
def load_vectorstore():
    try:
        session_id = request.form.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session ID'}), 400
        if session_id not in sessions:
            sessions[session_id] = Session(session_id)

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if not file.filename.endswith('.zip'):
            return jsonify({'error': 'File must be ZIP'}), 400

        safe_filename = secure_filename(f"upload_{session_id}_{file.filename}")
        zip_path = os.path.join(app.config['VECTORSTORE_FOLDER'], safe_filename)
        file.save(zip_path)

        session = sessions[session_id]
        success = session.vector_store.load_from_zip(zip_path)
        if success:
            return jsonify({'success': True, 'status': 'Vectorstore loaded successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to load vectorstore'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-chat', methods=['POST'])
def export_chat():
    try:
        session_id = request.json.get('session_id')
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session'}), 400
        session = sessions[session_id]
        chat_data = {'timestamp': datetime.now().isoformat(), 'session_id': session_id, 'conversation': session.chat_history.history}
        from io import BytesIO
        json_bytes = json.dumps(chat_data, indent=2).encode('utf-8')
        return send_file(BytesIO(json_bytes), as_attachment=True, download_name=f'chat_history_{session_id[:8]}.json', mimetype='application/json')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    try:
        session_id = request.json.get('session_id')
        if session_id in sessions:
            sessions[session_id].chat_history.clear()
            sessions[session_id].uploaded_files.clear()
            sessions[session_id].vector_store.clear_embed_progress()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== Start Server (HF compatible) ====================
def find_available_port(start_port=5000, end_port=5009):
    import socket
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available ports in range {start_port}-{end_port}")

if __name__ != '__main__':
    application = app
else:
    hf_port = os.environ.get('PORT')
    if hf_port:
        port = int(hf_port)
        host = '0.0.0.0'
        open_browser = False
        logger.info(f"Running on Hugging Face Space - port {port}")
    else:
        port = find_available_port(5000, 5009)
        host = '127.0.0.1'
        open_browser = True
        logger.info(f"Local development - using port {port}")

    logger.info("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   RAG Document Analyzer - v3.14 HF READY                         ║
    ║   Retrieval: HyDE + Multi-query + Sentence-window + Reranker     ║
    ║   Ready for Hugging Face Spaces deployment                       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    if open_browser:
        import webbrowser
        webbrowser.open(f'http://{host}:{port}')
    app.run(host=host, port=port, debug=False, use_reloader=False)