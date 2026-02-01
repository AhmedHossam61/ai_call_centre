# 🚀 Quick Start - Choose Your Version

## 📦 What You Have

You now have **TWO versions** of the Arabic RAG agent:

### 1️⃣ Simple Version (TF-IDF)
- **File:** `arabic_rag_agent.py`
- **Best for:** Small datasets, quick prototyping
- **Setup:** 2 minutes

### 2️⃣ Professional Version (ChromaDB)
- **File:** `arabic_rag_chromadb.py`
- **Best for:** Production, incremental updates, larger datasets
- **Setup:** 5 minutes

## 🤔 Which Should I Use?

### Choose **Simple Version** if:
- You have < 50 Q&A pairs
- You want the easiest setup
- You're just testing the concept

### Choose **ChromaDB Version** if:
- ✅ You have 50+ Q&A pairs
- ✅ You'll frequently add new questions
- ✅ You need persistent storage
- ✅ You want the best search quality
- ✅ You're building for production

**Recommended:** Start with ChromaDB version - it's production-ready!

---

## 🎯 Setup: ChromaDB Version (Recommended)

### Step 1: Install Dependencies

```bash
pip install chromadb google-generativeai pypdf python-docx --break-system-packages
```

### Step 2: Get API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy your key

### Step 3: Set API Key

```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Step 4: Run!

```bash
python arabic_rag_chromadb.py
```

**When prompted:**
- Document path: `sample_qa_arabic.docx`
- Force reload: `N`

### Step 5: Test Incremental Updates

```bash
# See it in action!
python demo_incremental_updates.py
```

---

## 🎯 Setup: Simple Version (Alternative)

### Step 1: Install Dependencies

```bash
pip install google-generativeai pypdf python-docx scikit-learn numpy --break-system-packages
```

### Step 2: Get API Key

Same as ChromaDB version above

### Step 3: Set API Key

```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Step 4: Run!

```bash
python arabic_rag_agent.py
```

---

## 📝 Using Your Own Q&A File

### Format Your Document

Create a Word document (`.docx`) or PDF with this format:

```
سؤال: ما هي ساعات العمل؟
جواب: نحن نعمل من الأحد إلى الخميس من الساعة 9 صباحاً حتى 5 مساءً.

سؤال: كيف يمكنني تتبع طلبي؟
جواب: يمكنك تتبع طلبي من خلال حسابك على الموقع.
```

**Also supports English:**
```
Q: What are your working hours?
A: We work Sunday to Thursday, 9 AM to 5 PM.
```

### Load Your File

When the agent asks for a file path, just enter it:
```
Enter path to Q&A document: /path/to/your/qa_file.docx
```

---

## 🧪 Test the Demo

### See Incremental Updates in Action

```bash
python demo_incremental_updates.py
```

This demonstrates:
1. ✅ First load: Encodes 5 questions
2. ✅ Reload: Skips encoding (uses cache)
3. ✅ Add 3 new: Only encodes the 3 new ones

---

## 📊 Example Chat Session

```
العميل: ما هي ساعات العمل؟
الموظف: نحن نعمل من الأحد إلى الخميس من الساعة 9 صباحاً حتى 5 مساءً. نحن مغلقون يومي الجمعة والسبت.

العميل: هل التوصيل مجاني؟
الموظف: نعم، نقدم توصيل مجاني للطلبات التي تزيد عن 200 ريال. للطلبات الأقل من ذلك، رسوم التوصيل 25 ريال.

العميل: stats
📊 Database Statistics:
  Total Q&A pairs: 10
  Documents tracked: 1
  Last updated: 2025-01-31T10:30:00
  Database path: ./chroma_db

العميل: quit
شكراً لاستخدامك الوكيل الذكي!
```

---

## 🔄 Adding New Questions (ChromaDB Version)

### Step 1: Edit Your Document

Open `sample_qa_arabic.docx` and add:

```
سؤال: كيف أتواصل مع الدعم الفني؟
جواب: يمكنك التواصل معنا عبر الرقم 920001234 أو البريد الإلكتروني.
```

### Step 2: Reload

```bash
python arabic_rag_chromadb.py
```

**Output:**
```
Found 11 Q&A pairs in document
Processing 1 new Q&A pairs...
Encoding 1 new Q&A pairs...
✓ Added 1 new embeddings to database
✓ Total Q&A pairs in database: 11
```

**Magic!** Only encoded the 1 new question! 🎉

---

## 📁 Files Explained

### Core Files

| File | Purpose |
|------|---------|
| `arabic_rag_chromadb.py` | **Main agent** with ChromaDB (RECOMMENDED) |
| `arabic_rag_agent.py` | Simple version with TF-IDF |
| `sample_qa_arabic.docx` | Example Q&A document in Arabic |

### Documentation

| File | Purpose |
|------|---------|
| `README_CHROMADB.md` | Full documentation for ChromaDB version |
| `README.md` | Documentation for simple version |
| `VERSION_COMPARISON.md` | Compare both versions |
| `QUICKSTART_MASTER.md` | This file! |

### Scripts & Tools

| File | Purpose |
|------|---------|
| `demo_incremental_updates.py` | See incremental updates in action |
| `test_setup.py` | Test your setup |
| `examples.py` | Code examples |

### Requirements

| File | Purpose |
|------|---------|
| `requirements_chromadb.txt` | Dependencies for ChromaDB version |
| `requirements.txt` | Dependencies for simple version |

---

## 🆘 Troubleshooting

### Problem: "chromadb not installed"

```bash
pip install chromadb --break-system-packages
```

### Problem: "GEMINI_API_KEY not set"

```bash
export GEMINI_API_KEY="your-key-here"
```

Or enter it when the script asks.

### Problem: "No Q&A pairs found"

Check your document format:
- Must have `سؤال:` or `Q:` markers
- Must have `جواب:` or `A:` markers
- See `sample_qa_arabic.docx` for examples

### Problem: Slow encoding

**If using Gemini embeddings:**
- Normal! First time takes 10-20 seconds for 100 questions
- Subsequent loads are instant (cached)

**To speed up:**
- Don't set `GEMINI_API_KEY` to use default embeddings (faster but lower quality)

---

## 🎓 Next Steps

1. ✅ Run the demo: `python demo_incremental_updates.py`
2. ✅ Try with sample data: Use `sample_qa_arabic.docx`
3. ✅ Create your own Q&A document
4. ✅ Test incremental updates
5. ✅ Integrate into your call center system

---

## 🌟 Key Features Summary

### ChromaDB Version Has:
- ✅ **Persistent storage** - survives restarts
- ✅ **Incremental updates** - only encode new Q&A
- ✅ **Better search** - semantic understanding
- ✅ **Database stats** - track your data
- ✅ **Production ready** - built for scale

### Both Versions Support:
- ✅ **Arabic & English** Q&A documents
- ✅ **DOCX & PDF** files
- ✅ **Natural conversations** with Gemini
- ✅ **Easy to use** - simple CLI interface

---

## 📞 Ready for Voice?

This RAG system is designed to be extended with voice capabilities:

**Phase 2 (Future):**
- Speech-to-text for customer input
- Text-to-speech for agent responses
- Telephony integration
- Real-time conversation

The current chat interface can easily be replaced with voice I/O!

---

## 🎯 Your Action Plan

```bash
# 1. Install ChromaDB version
pip install -r requirements_chromadb.txt --break-system-packages

# 2. Get API key
# Visit: https://makersuite.google.com/app/apikey

# 3. Set API key
export GEMINI_API_KEY="your-key"

# 4. Run demo
python demo_incremental_updates.py

# 5. Try with your data
python arabic_rag_chromadb.py
```

**That's it! You're ready to go! 🚀**

---

**Questions?** Read the full documentation:
- `README_CHROMADB.md` - Complete ChromaDB guide
- `VERSION_COMPARISON.md` - Compare versions
- Or just run the demo and explore!

**Happy chatting!** 🤖💬
