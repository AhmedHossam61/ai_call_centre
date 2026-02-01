"""
Dialect Detection using Gemini 2.5 Flash
Analyzes Arabic text to identify dialect without training
"""

import json
from typing import Tuple

class DialectDetector:
    """
    LLM-based dialect detector using Gemini
    Supports: Egyptian, Gulf, Levantine, Moroccan, MSA
    """
    
    DIALECT_PROMPT = """أنت خبير في اللهجات العربية. قم بتحليل النص التالي وحدد اللهجة المستخدمة بدقة.

اللهجات المدعومة:
- مصري (Egyptian) - استخدام: ازيك، عايز، ايه، دا/دي، انت/انتي، معلش، عامل ايه
- خليجي (Gulf/Khaleeji) - استخدام: شلونك، شنو، ويش، عساك، يالله، تبغى
- شامي (Levantine) - استخدام: كيفك، شو، هيك، منيح، يلا، بدك
- مغربي (Moroccan/Maghrebi) - استخدام: كيفاش، واش، بزاف، مزيان
- فصحى (Modern Standard Arabic) - استخدام: كيف حالك، ماذا، هذا، ذلك

النص المراد تحليله: "{text}"

قم بالتحليل بناءً على:
1. الكلمات والمفردات الخاصة باللهجة
2. التراكيب اللغوية والنحوية
3. الأفعال والضمائر المستخدمة

أجب بصيغة JSON فقط، بدون أي نص إضافي أو علامات markdown:
{{
    "dialect": "اسم اللهجة بالإنجليزية (egyptian/gulf/levantine/moroccan/msa)",
    "confidence": "رقم من 0.0 إلى 1.0",
    "reasoning": "سبب قصير للاختيار مع ذكر الكلمات الدالة"
}}"""
    
    def __init__(self, model):
        """
        Initialize dialect detector
        
        Args:
            model: Gemini model instance
        """
        self.model = model
        self.supported_dialects = ['egyptian', 'gulf', 'levantine', 'moroccan', 'msa']
    
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect Arabic dialect from transcribed text
        
        Args:
            text: Transcribed Arabic text
        
        Returns:
            Tuple of (dialect_name, confidence_score)
        """
        if not text or len(text.strip()) < 3:
            return 'msa', 0.3  # Default for very short text
        
        prompt = self.DIALECT_PROMPT.format(text=text)
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean up response - remove markdown code blocks if present
            result_text = self._clean_json_response(result_text)
            
            # Parse JSON
            result = json.loads(result_text)
            
            dialect = result.get('dialect', 'msa').lower()
            confidence = float(result.get('confidence', 0.5))
            reasoning = result.get('reasoning', 'N/A')
            
            # Validate dialect
            if dialect not in self.supported_dialects:
                print(f"⚠️  Unknown dialect '{dialect}', defaulting to MSA")
                dialect = 'msa'
                confidence = 0.5
            
            print(f"🔍 Detected: {dialect} (confidence: {confidence:.2f})")
            print(f"   Reasoning: {reasoning}")
            
            return dialect, confidence
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"   Raw response: {result_text[:100]}...")
            return 'msa', 0.5
            
        except Exception as e:
            print(f"❌ Dialect detection error: {e}")
            return 'msa', 0.5
    
    def _clean_json_response(self, text: str) -> str:
        """Remove markdown formatting from JSON response"""
        # Remove ```json and ``` markers
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        return text.strip()
    
    def batch_detect(self, texts: list) -> list:
        """
        Detect dialect for multiple texts
        Useful for analyzing conversation patterns
        
        Args:
            texts: List of Arabic texts
        
        Returns:
            List of (dialect, confidence) tuples
        """
        results = []
        for text in texts:
            dialect, confidence = self.detect(text)
            results.append((dialect, confidence))
        
        return results
