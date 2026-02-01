#!/usr/bin/env python3
"""
Arabic RAG Agent with ChromaDB Vector Database
Supports incremental updates - only encodes new Q&A pairs
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


class ArabicDocumentReader:
    """Read Arabic text from DOCX or PDF files"""
    
    @staticmethod
    def read_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except ImportError:
            print("Error: python-docx not installed. Run: pip install python-docx --break-system-packages")
            return ""
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""
    
    @staticmethod
    def read_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            print("Error: pypdf not installed. Run: pip install pypdf --break-system-packages")
            return ""
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def read_document(file_path: str) -> str:
        """Read document based on file extension"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        
        if ext == '.docx':
            return ArabicDocumentReader.read_docx(file_path)
        elif ext == '.pdf':
            return ArabicDocumentReader.read_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use .docx or .pdf")


class ChromaRAG:
    """RAG system using ChromaDB with incremental updates"""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "qa_collection"):
        """Initialize ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb not installed.\n"
                "Run: pip install chromadb --break-system-packages"
            )
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = collection_name
        self.db_path = db_path
        self.metadata_file = os.path.join(db_path, "metadata.json")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Load metadata (tracks what's been encoded)
        self.metadata = self._load_metadata()
        
        print(f"ChromaDB initialized at: {db_path}")
        print(f"Collection: {collection_name}")
        print(f"Existing Q&A pairs: {self.collection.count()}")
    
    def _load_metadata(self) -> Dict:
        """Load metadata about encoded documents"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"documents": {}, "last_updated": None}
    
    def _save_metadata(self):
        """Save metadata to disk"""
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file content to detect changes"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _generate_qa_id(self, question: str, index: int) -> str:
        """Generate unique ID for Q&A pair"""
        # Use hash of question + index for uniqueness
        content = f"{question}_{index}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
    
    def parse_qa_document(self, text: str) -> List[Dict[str, str]]:
        """
        Parse Q&A document into question-answer pairs.
        Expected format:
        سؤال: Question text?
        جواب: Answer text.
        """
        qa_pairs = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        current_q = None
        current_a = None
        
        for line in lines:
            # Check for question markers (English and Arabic)
            if line.startswith('Q:') or line.startswith('q:') or \
               line.startswith('سؤال:') or line.startswith('Question:'):
                if current_q and current_a:
                    qa_pairs.append({'question': current_q, 'answer': current_a})
                current_q = line.split(':', 1)[1].strip()
                current_a = None
            
            # Check for answer markers
            elif line.startswith('A:') or line.startswith('a:') or \
                 line.startswith('جواب:') or line.startswith('إجابة:') or \
                 line.startswith('Answer:'):
                current_a = line.split(':', 1)[1].strip()
            
            # Continue previous answer if it exists
            elif current_a is not None and not line.startswith(('Q:', 'q:', 'سؤال:', 'Question:')):
                current_a += ' ' + line
        
        # Add last pair
        if current_q and current_a:
            qa_pairs.append({'question': current_q, 'answer': current_a})
        
        return qa_pairs
    
    def load_from_file(self, file_path: str, embedding_function=None, force_reload: bool = False):
        """
        Load Q&A pairs from document with incremental updates
        
        Args:
            file_path: Path to DOCX or PDF file
            embedding_function: Function to generate embeddings (uses Gemini if None)
            force_reload: If True, re-encode everything even if unchanged
        """
        print(f"\nLoading document: {file_path}")
        
        # Read document
        text = ArabicDocumentReader.read_document(file_path)
        if not text.strip():
            raise ValueError("No text extracted from document")
        
        # Parse Q&A pairs
        qa_pairs = self.parse_qa_document(text)
        if len(qa_pairs) == 0:
            raise ValueError("No Q&A pairs found in document. Check format.")
        
        print(f"Found {len(qa_pairs)} Q&A pairs in document")
        
        # Check if document has changed
        file_hash = self._get_file_hash(file_path)
        file_name = os.path.basename(file_path)
        
        if not force_reload and file_name in self.metadata["documents"]:
            stored_hash = self.metadata["documents"][file_name].get("hash")
            if stored_hash == file_hash:
                print("✓ Document unchanged - using existing embeddings")
                print(f"✓ Total Q&A pairs in database: {self.collection.count()}")
                return
        
        # Determine what's new
        existing_count = self.metadata["documents"].get(file_name, {}).get("count", 0)
        new_pairs = qa_pairs[existing_count:]
        
        if len(new_pairs) == 0 and not force_reload:
            print("✓ No new Q&A pairs to add")
            return
        
        print(f"Processing {len(new_pairs)} new Q&A pairs...")
        
        # Get embedding function
        if embedding_function is None:
            embedding_function = self._get_default_embedding_function()
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, qa in enumerate(new_pairs, start=existing_count):
            qa_id = self._generate_qa_id(qa['question'], i)
            
            # Combine question and answer for embedding
            # This helps with semantic search
            combined_text = f"سؤال: {qa['question']}\nجواب: {qa['answer']}"
            
            documents.append(combined_text)
            metadatas.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'index': i,
                'source': file_name,
                'added_at': datetime.now().isoformat()
            })
            ids.append(qa_id)
        
        # Add to ChromaDB (will automatically embed)
        if documents:
            print(f"Encoding {len(documents)} new Q&A pairs...")
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✓ Added {len(documents)} new embeddings to database")
        
        # Update metadata
        self.metadata["documents"][file_name] = {
            "hash": file_hash,
            "count": len(qa_pairs),
            "last_updated": datetime.now().isoformat()
        }
        self.metadata["last_updated"] = datetime.now().isoformat()
        self._save_metadata()
        
        print(f"✓ Total Q&A pairs in database: {self.collection.count()}")
    
    def _get_default_embedding_function(self):
        """Get default embedding function using Gemini"""
        try:
            import google.generativeai as genai
            
            # Use Gemini's embedding model
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("Warning: GEMINI_API_KEY not set, using ChromaDB's default embeddings")
                return None
            
            # Return a function that uses Gemini embeddings
            def embed_texts(texts):
                genai.configure(api_key=api_key)
                embeddings = []
                for text in texts:
                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result['embedding'])
                return embeddings
            
            return embed_texts
            
        except ImportError:
            print("Warning: google-generativeai not installed, using ChromaDB's default embeddings")
            return None
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, str], float]]:
        """
        Retrieve most relevant Q&A pairs
        
        Returns:
            List of (qa_dict, similarity_score) tuples
        """
        if self.collection.count() == 0:
            return []
        
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count())
        )
        
        # Format results
        retrieved = []
        if results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                qa = {
                    'question': metadata['question'],
                    'answer': metadata['answer']
                }
                # ChromaDB returns distances (lower is better), convert to similarity
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                
                retrieved.append((qa, similarity))
        
        return retrieved
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            "total_qa_pairs": self.collection.count(),
            "documents_tracked": len(self.metadata["documents"]),
            "last_updated": self.metadata.get("last_updated"),
            "db_path": self.db_path
        }
    
    def reset_database(self):
        """Clear all data and start fresh"""
        print("Resetting database...")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.metadata = {"documents": {}, "last_updated": None}
        self._save_metadata()
        print("✓ Database reset complete")


class ArabicCallCenterAgent:
    """Arabic call center agent with ChromaDB vector database"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash", 
                 db_path: str = "./chroma_db"):
        """Initialize the agent with Gemini API and ChromaDB"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai not installed.\n"
                "Run: pip install google-generativeai --break-system-packages\n"
                "Or get it from: https://pypi.org/project/google-generativeai/"
            )
        
        self.rag = ChromaRAG(db_path=db_path)
        self.chat_history = []
    
    def load_knowledge_base(self, file_path: str, force_reload: bool = False):
        """
        Load Q&A knowledge base with incremental updates
        
        Args:
            file_path: Path to DOCX or PDF file
            force_reload: If True, re-encode everything
        """
        self.rag.load_from_file(file_path, force_reload=force_reload)
    
    def get_response(self, user_query: str) -> str:
        """Get response using RAG + Gemini"""
        
        # Retrieve relevant Q&A pairs
        relevant_qa = self.rag.retrieve(user_query, top_k=3)
        
        # Build context from retrieved Q&A
        context = ""
        if relevant_qa:
            context = "المعلومات المتاحة من قاعدة المعرفة:\n\n"
            for i, (qa, score) in enumerate(relevant_qa, 1):
                context += f"سؤال {i}: {qa['question']}\n"
                context += f"إجابة {i}: {qa['answer']}\n\n"
        
        # Build prompt for Gemini
        prompt = f"""أنت موظف خدمة عملاء في مركز اتصال. مهمتك الإجابة على أسئلة العملاء بشكل مهني ومفيد باللغة العربية.

{context}

سؤال العميل: {user_query}

قدم إجابة واضحة ومفيدة بناءً على المعلومات المتاحة. إذا لم تجد معلومات كافية في قاعدة المعرفة، أخبر العميل بأدب أنك ستحول السؤال للمختصين.

الإجابة:"""
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            answer = response.text
            
            # Store in history
            self.chat_history.append({
                'user': user_query,
                'assistant': answer,
                'retrieved_qa': relevant_qa
            })
            
            return answer
        
        except Exception as e:
            return f"عذراً، حدث خطأ: {str(e)}"
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return self.rag.get_stats()
    
    def reset_database(self):
        """Reset the vector database"""
        self.rag.reset_database()


def main():
    """Main function to run the chat interface"""
    
    print("=" * 60)
    print("وكيل مركز الاتصال الذكي مع ChromaDB")
    print("Arabic Call Center Agent with Vector Database")
    print("=" * 60)
    print()
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set your Gemini API key:")
        api_key = input("GEMINI_API_KEY: ").strip()
        if not api_key:
            print("Error: API key is required")
            sys.exit(1)
    
    # Get knowledge base file
    kb_file = input("\nEnter path to Q&A document (DOCX or PDF): ").strip()
    
    if not kb_file:
        print("Error: Knowledge base file is required")
        sys.exit(1)
    
    # Ask about force reload
    force_reload = input("\nForce re-encode everything? (y/N): ").strip().lower() == 'y'
    
    # Initialize agent
    try:
        print("\nInitializing agent with ChromaDB...")
        agent = ArabicCallCenterAgent(api_key)
        agent.load_knowledge_base(kb_file, force_reload=force_reload)
        
        # Show stats
        stats = agent.get_stats()
        print(f"\nDatabase Stats:")
        print(f"  - Total Q&A pairs: {stats['total_qa_pairs']}")
        print(f"  - Documents tracked: {stats['documents_tracked']}")
        print(f"  - Last updated: {stats.get('last_updated', 'Never')}")
        
    except Exception as e:
        print(f"Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Agent ready! Type 'quit' or 'exit' to end")
    print("Type 'stats' to see database statistics")
    print("=" * 60)
    print()
    
    # Chat loop
    while True:
        try:
            user_input = input("\nالعميل: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'خروج', 'إنهاء']:
                print("\nشكراً لاستخدامك الوكيل الذكي!")
                break
            
            if user_input.lower() == 'stats':
                stats = agent.get_stats()
                print("\n📊 Database Statistics:")
                print(f"  Total Q&A pairs: {stats['total_qa_pairs']}")
                print(f"  Documents tracked: {stats['documents_tracked']}")
                print(f"  Last updated: {stats.get('last_updated', 'Never')}")
                print(f"  Database path: {stats['db_path']}")
                continue
            
            print("\nالموظف: ", end="", flush=True)
            response = agent.get_response(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nشكراً لاستخدامك الوكيل الذكي!")
            break
        except Exception as e:
            print(f"\nخطأ: {e}")


if __name__ == "__main__":
    main()
