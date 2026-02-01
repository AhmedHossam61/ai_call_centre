# Arabic RAG Call Center Agent with ChromaDB

A production-ready RAG agent with **vector database persistence** and **incremental updates**. Only encodes new Q&A pairs, never re-encodes existing data.

## 🚀 New Features (ChromaDB Version)

### ✅ Vector Database with Persistence
- Uses **ChromaDB** for fast semantic search
- **Persistent storage** - embeddings saved to disk
- **Survives restarts** - no need to re-encode on every run

### ✅ Incremental Updates
- **Smart encoding**: Only encodes NEW Q&A pairs
- **Tracks changes**: Detects when documents are modified
- **Saves time**: Add question 11 without re-encoding questions 1-10

### ✅ Better Embeddings
- Uses **Gemini embeddings** for semantic search (optional)
- Falls back to ChromaDB's default embeddings
- Much better than TF-IDF for Arabic

### ✅ Production Ready
- Metadata tracking
- Database statistics
- Reset functionality
- Error handling

## 📊 How Incremental Updates Work

```
First Load (10 questions):
  → Encodes all 10 questions
  → Saves to ChromaDB
  → Creates metadata file

Second Load (same file):
  → Detects file unchanged
  → Skips encoding ✓
  → Uses existing embeddings

Add Questions 11-15:
  → Detects 5 new questions
  → Only encodes the 5 new ones ✓
  → Total: 15 questions in database
```

## 🔧 Installation

### 1. Install Dependencies

```bash
pip install chromadb google-generativeai pypdf python-docx --break-system-packages
```

Or use the requirements file:

```bash
pip install -r requirements_chromadb.txt --break-system-packages
```

### 2. Get Gemini API Key

Get your free API key from: https://makersuite.google.com/app/apikey

### 3. Set Environment Variable

```bash
export GEMINI_API_KEY="your-api-key-here"
```

## 🎯 Quick Start

### Basic Usage

```bash
python arabic_rag_chromadb.py
```

When prompted:
1. Enter your API key (if not set as environment variable)
2. Enter path to your Q&A document: `sample_qa_arabic.docx`
3. Choose whether to force re-encode (usually 'N')

### First Time Run

```
Enter path to Q&A document: sample_qa_arabic.docx
Force re-encode everything? (y/N): N

ChromaDB initialized at: ./chroma_db
Collection: qa_collection
Existing Q&A pairs: 0

Found 10 Q&A pairs in document
Processing 10 new Q&A pairs...
Encoding 10 new Q&A pairs...
✓ Added 10 new embeddings to database
✓ Total Q&A pairs in database: 10
```

### Second Run (Same File)

```
Found 10 Q&A pairs in document
✓ Document unchanged - using existing embeddings
✓ Total Q&A pairs in database: 10
```

### After Adding Questions

```
Found 15 Q&A pairs in document
Processing 5 new Q&A pairs...
Encoding 5 new Q&A pairs...
✓ Added 5 new embeddings to database
✓ Total Q&A pairs in database: 15
```

## 📝 Q&A Document Format

Same as before - supports both Arabic and English markers:

```
سؤال: ما هي ساعات العمل؟
جواب: نحن نعمل من الأحد إلى الخميس من الساعة 9 صباحاً حتى 5 مساءً.

سؤال: كيف يمكنني تتبع طلبي؟
جواب: يمكنك تتبع طلبك من خلال الدخول إلى حسابك.
```

## 💻 Programmatic Usage

### Example 1: Basic Usage

```python
from arabic_rag_chromadb import ArabicCallCenterAgent

# Initialize agent
agent = ArabicCallCenterAgent(
    api_key="your-api-key",
    db_path="./my_database"  # Where to store embeddings
)

# Load knowledge base (incremental)
agent.load_knowledge_base("qa_file.docx")

# Get response
response = agent.get_response("ما هي ساعات العمل؟")
print(response)

# Check stats
stats = agent.get_stats()
print(f"Total Q&A: {stats['total_qa_pairs']}")
```

### Example 2: Force Reload

```python
# Force re-encode everything (use when document structure changes)
agent.load_knowledge_base("qa_file.docx", force_reload=True)
```

### Example 3: Multiple Documents

```python
# Load multiple Q&A files
agent.load_knowledge_base("general_qa.docx")
agent.load_knowledge_base("technical_qa.docx")
agent.load_knowledge_base("billing_qa.docx")

# All Q&A pairs are now in the same database
stats = agent.get_stats()
print(f"Total documents: {stats['documents_tracked']}")
print(f"Total Q&A pairs: {stats['total_qa_pairs']}")
```

### Example 4: Using ChromaDB Directly

```python
from arabic_rag_chromadb import ChromaRAG

# Initialize database
rag = ChromaRAG(db_path="./chroma_db")

# Load document
rag.load_from_file("qa.docx")

# Retrieve without Gemini
results = rag.retrieve("ما هي ساعات العمل؟", top_k=5)
for qa, score in results:
    print(f"Score: {score:.3f}")
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}")
```

## 🧪 Demo Script

Run the demo to see incremental updates in action:

```bash
python demo_incremental_updates.py
```

This will:
1. Create a Q&A file with 5 questions
2. Load it (encodes all 5)
3. Reload it (skips encoding)
4. Add 3 more questions
5. Reload again (only encodes the 3 new ones)

## 📂 File Structure

```
your-project/
├── arabic_rag_chromadb.py      # Main agent with ChromaDB
├── sample_qa_arabic.docx        # Sample Q&A file
├── demo_incremental_updates.py  # Demo script
├── requirements_chromadb.txt    # Dependencies
├── chroma_db/                   # Vector database (created automatically)
│   ├── chroma.sqlite3          # ChromaDB storage
│   ├── metadata.json           # Tracks what's encoded
│   └── ...
└── demo_chroma_db/             # Demo database (from demo script)
```

## 🎛️ Advanced Configuration

### Custom Database Path

```python
agent = ArabicCallCenterAgent(
    api_key="key",
    db_path="/path/to/your/database"
)
```

### Custom Collection Name

```python
from arabic_rag_chromadb import ChromaRAG

rag = ChromaRAG(
    db_path="./chroma_db",
    collection_name="my_custom_collection"
)
```

### Different Gemini Models

```python
# Use Gemini Pro for better quality
agent = ArabicCallCenterAgent(
    api_key="key",
    model_name="gemini-1.5-pro"
)

# Use Flash for speed (default)
agent = ArabicCallCenterAgent(
    api_key="key",
    model_name="gemini-1.5-flash"
)
```

### Reset Database

```python
# Clear all embeddings and start fresh
agent.reset_database()
```

## 📊 Database Statistics

Get detailed stats about your database:

```python
stats = agent.get_stats()

print(stats)
# {
#   'total_qa_pairs': 15,
#   'documents_tracked': 2,
#   'last_updated': '2025-01-31T10:30:00',
#   'db_path': './chroma_db'
# }
```

Or in the chat interface, type: `stats`

## 🔍 How It Works

### 1. Document Hashing
- Calculates MD5 hash of the document
- Compares with stored hash to detect changes
- Skips encoding if hash matches

### 2. Metadata Tracking
- Stores document hash and Q&A count
- Tracks which questions have been encoded
- Saves metadata to `metadata.json`

### 3. Incremental Encoding
- Compares current Q&A count with stored count
- Only processes new pairs: `new_pairs = all_pairs[existing_count:]`
- Adds only new embeddings to ChromaDB

### 4. Persistent Storage
- ChromaDB saves embeddings to disk
- Survives program restarts
- No need to re-encode on next run

## ⚡ Performance

### Encoding Speed
- First load (10 Q&A): ~5-10 seconds (with Gemini embeddings)
- Reload (unchanged): <1 second (skipped)
- Add 5 new Q&A: ~2-5 seconds (only encodes 5)

### Search Speed
- Query time: <100ms for most queries
- Scales well to thousands of Q&A pairs
- ChromaDB uses HNSW for fast similarity search

## 🆚 Comparison: Old vs New

| Feature | Old (TF-IDF) | New (ChromaDB) |
|---------|-------------|----------------|
| Persistence | ❌ None | ✅ Disk storage |
| Incremental | ❌ Re-encode all | ✅ Only new ones |
| Embeddings | TF-IDF (basic) | Gemini (semantic) |
| Speed | Fast but limited | Fast & scalable |
| Quality | Good | Excellent |
| Restart | Re-encode all | Load from disk |

## 🐛 Troubleshooting

### ChromaDB Not Installing

```bash
# Try with --no-cache-dir
pip install chromadb --no-cache-dir --break-system-packages
```

### Embeddings Taking Too Long

```python
# Use ChromaDB's default embeddings (faster, no API calls)
# Just don't set GEMINI_API_KEY
```

### Database Corrupted

```python
# Reset and start fresh
agent.reset_database()
agent.load_knowledge_base("qa.docx", force_reload=True)
```

### File Not Detected as Changed

```python
# Force reload
agent.load_knowledge_base("qa.docx", force_reload=True)
```

## 🎓 Best Practices

1. **Don't force reload** unless necessary (wastes time & API calls)
2. **Keep backups** of your `chroma_db` folder
3. **Use meaningful collection names** for different Q&A sets
4. **Monitor your Gemini API usage** (check quotas)
5. **Add questions at the end** of your document for best performance

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Voice integration (speech-to-text & text-to-speech)
- [ ] Web dashboard for managing Q&A
- [ ] Analytics and conversation tracking
- [ ] Auto-update from Google Sheets
- [ ] Clustering similar questions
- [ ] A/B testing different prompts

## 📄 License

Free to use and modify for your call center needs.

## 🤝 Support

For issues:
- Check the demo script: `python demo_incremental_updates.py`
- Review the troubleshooting section
- Inspect the metadata file: `cat chroma_db/metadata.json`

---

**Built with:** ChromaDB, Gemini API, Python
**Perfect for:** Arabic call centers, customer support, FAQ systems
**Ready for:** Voice integration in Phase 2! 🎤
