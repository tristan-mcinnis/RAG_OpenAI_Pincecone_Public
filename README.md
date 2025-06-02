# RAG System Usage Guide

## 🚀 Quick Start Sequence

### 1. **Health Check** (Recommended First Step)
```bash
python main.py health_check
```

### 2. **Process File and Query** (Most Common)
```bash
# Single file
python main.py process_query "document.txt" "What is the main topic?"
python main.py process_query "/path/to/transcript.txt" "What do participants think about basketball?"

# Directory with multiple files
python main.py process_query "/path/to/documents/" "What are the key findings?"
```

### 3. **Process Files Only** (Index Without Querying)
```bash
# Single file
python main.py process "document.txt"

# Multiple files
python main.py process "/path/to/documents/"
```

### 4. **Query Existing Data** (No File Processing)
```bash
python main.py query "What are the main themes?"
python main.py query "What did participants say about sports?"
```

## 🔍 Document Retrieval (No AI Response)

### Basic Retrieval
```bash
python main.py retrieve "basketball opinions"
python main.py retrieve "artificial intelligence" --top-k 20
```

### Advanced Retrieval
```bash
# High-confidence matches only
python main.py retrieve "machine learning" --min-score 0.6

# Research mode: find all mentions
python main.py retrieve "neural networks" --top-k 50 --min-score 0.2
```

## 💬 Verbatim Extraction (Research Quotes)

### Basic Verbatim Extraction
```bash
python main.py extract_verbatims "basketball opinions"
python main.py extract_verbatims "devin booker"
```

### Filtered Verbatim Extraction
```bash
# Length filters
python main.py extract_verbatims "devin booker" --min-length 30 --max-length 200

# Exclude moderator quotes
python main.py extract_verbatims "game analysis" --exclude-moderator

# Include moderator quotes
python main.py extract_verbatims "game analysis" --include-moderator

# Filter by participant demographics
python main.py extract_verbatims "sports preferences" --participant-filter "M, 18-24" --top-k 30
```

### Verbatim Output Formats
```bash
# Research format (default): "Quote" - Speaker, Location, Demographics
python main.py extract_verbatims "basketball" --format research

# Quotes only: "Quote"
python main.py extract_verbatims "player opinions" --format quotes_only --min-length 50

# Detailed format (with metadata)
python main.py extract_verbatims "game analysis" --format detailed

# CSV format
python main.py extract_verbatims "basketball" --format csv

# Export to CSV file
python main.py extract_verbatims "basketball" --format csv --export-csv basketball_quotes.csv
```

## 🎮 Interactive Mode

### Start Interactive Session
```bash
python main.py interactive
```

**Interactive Flow:**
1. Enter file/directory path when prompted
2. Ask multiple questions
3. Type 'exit' or 'quit' to end session

## ⚙️ Command Arguments

### Global Options
| Argument | Description | Default |
|----------|-------------|---------|
| `--top-k` | Number of documents to retrieve | 10-20 |
| `--min-score` | Minimum similarity score | 0.1 |

### Verbatim-Specific Options
| Argument | Description | Default |
|----------|-------------|---------|
| `--min-length` | Minimum quote length (characters) | 20 |
| `--max-length` | Maximum quote length (characters) | 500 |
| `--exclude-moderator` | Exclude moderator quotes | True |
| `--include-moderator` | Include moderator quotes | False |
| `--participant-filter` | Filter by demographics (e.g., "M, 18-24") | None |
| `--format` | Output format (research/quotes_only/detailed/csv) | research |
| `--export-csv` | Export to CSV file | None |

## 📋 Complete Command Reference

```bash
# Health and system
python main.py health_check

# File processing
python main.py process "/path/to/files/"
python main.py process_query "/path/to/files/" "question"

# Querying
python main.py query "question"
python main.py retrieve "search terms" --top-k 20 --min-score 0.3

# Verbatim extraction
python main.py extract_verbatims "topic" --min-length 30 --exclude-moderator --format research

# Interactive mode
python main.py interactive

# System management
python main.py delete_all  # ⚠️ Dangerous: deletes all data
```
