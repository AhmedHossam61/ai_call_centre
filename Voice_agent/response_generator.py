"""
Response Generation in Detected Dialect
Uses Gemini 2.5 Flash to generate natural responses
"""

class ResponseGenerator:
    """
    Generates customer service responses in the detected dialect
    """
    
    # Dialect names in Arabic
    DIALECT_NAMES = {
        'egyptian': 'المصرية',
        'gulf': 'الخليجية', 
        'levantine': 'الشامية',
        'moroccan': 'المغربية',
        'msa': 'العربية الفصحى'
    }
    
    # Example phrases for each dialect
    DIALECT_EXAMPLES = {
        'egyptian': '''أمثلة:
- "ازيك يا فندم؟ انت عايز ايه النهاردة؟"
- "تمام، هساعدك دلوقتي"
- "معلش على الإزعاج"''',
        
        'gulf': '''أمثلة:
- "شلونك؟ شنو تبغى اليوم؟"
- "تمام، بساعدك الحين"
- "عساك على القوة"''',
        
        'levantine': '''أمثلة:
- "كيفك؟ شو بدك اليوم؟"
- "تمام، رح ساعدك هلأ"
- "يلا ما في مشكلة"''',
        
        'moroccan': '''أمثلة:
- "كيفاش راك؟ واش بغيت اليوم؟"
- "مزيان، غادي نعاونك دابا"
- "ماشي مشكل"''',
        
        'msa': '''أمثلة:
- "كيف حالك؟ ماذا تريد اليوم؟"
- "حسناً، سأساعدك الآن"
- "لا بأس"'''
    }
    
    def __init__(self, model):
        """
        Initialize response generator
        
        Args:
            model: Gemini model instance
        """
        self.model = model
    
    def generate(self, user_query: str, dialect: str, context: list = None, 
                 system_context: str = None) -> str:
        """
        Generate customer service response in specified dialect
        
        Args:
            user_query: Customer's question/request
            dialect: Detected dialect (egyptian, gulf, levantine, moroccan, msa)
            context: Recent conversation history (list of dicts)
            system_context: Optional business context (service info, policies, etc.)
        
        Returns:
            Response text in the specified dialect
        """
        dialect_name = self.DIALECT_NAMES.get(dialect, 'العربية الفصحى')
        dialect_examples = self.DIALECT_EXAMPLES.get(dialect, '')
        
        # Build conversation context
        context_text = self._build_context(context)
        
        # Build system context
        business_context = system_context or "أنت مساعد خدمة عملاء محترف في مركز اتصال."
        
        # Create prompt
        prompt = f"""{business_context}

يجب أن تتحدث حصرياً باللهجة {dialect_name}.

{dialect_examples}

قواعد مهمة:
1. استخدم فقط مفردات وتعبيرات اللهجة {dialect_name}
2. كن ودوداً ومحترفاً ومتعاوناً
3. أجب بإيجاز ووضوح (2-3 جمل)
4. لا تستخدم الفصحى أو لهجات أخرى نهائياً
5. ركز على حل مشكلة العميل

{context_text}

سؤال/طلب العميل الحالي: {user_query}

أجب الآن فقط باللهجة {dialect_name}:"""
        
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            # Validate response is not empty
            if not answer or len(answer) < 5:
                return self._get_fallback_response(dialect)
            
            return answer
            
        except Exception as e:
            print(f"❌ Response generation error: {e}")
            return self._get_fallback_response(dialect)
    
    def _build_context(self, context: list) -> str:
        """Build conversation context string"""
        if not context or len(context) == 0:
            return ""
        
        context_text = "المحادثة السابقة:\n"
        for interaction in context:
            context_text += f"العميل: {interaction['user']}\n"
            context_text += f"المساعد: {interaction['assistant']}\n"
        
        return context_text
    
    def _get_fallback_response(self, dialect: str) -> str:
        """Get fallback response if generation fails"""
        fallbacks = {
            'egyptian': "عذراً، ممكن تعيد السؤال تاني؟",
            'gulf': "عذراً، ممكن تعيد السؤال؟",
            'levantine': "عذراً، ممكن تعيد السؤال؟",
            'moroccan': "سمح لي، ممكن تعاود السؤال؟",
            'msa': "عذراً، هل يمكنك إعادة السؤال؟"
        }
        return fallbacks.get(dialect, "عذراً، حدث خطأ. كيف يمكنني مساعدتك؟")
    
    def generate_greeting(self, dialect: str) -> str:
        """Generate initial greeting in dialect"""
        greetings = {
            'egyptian': "أهلاً بيك! ازيك؟ أقدر أساعدك في ايه النهاردة؟",
            'gulf': "مرحباً! شلونك؟ شنو أقدر أساعدك فيه اليوم؟",
            'levantine': "أهلاً فيك! كيفك؟ شو فيني ساعدك فيه اليوم؟",
            'moroccan': "مرحبا بيك! كيفاش راك؟ فاش نقدر نعاونك اليوم؟",
            'msa': "مرحباً بك! كيف يمكنني مساعدتك اليوم؟"
        }
        return greetings.get(dialect, greetings['msa'])
    
    def generate_closing(self, dialect: str) -> str:
        """Generate call closing in dialect"""
        closings = {
            'egyptian': "شكراً ليك! في أي خدمة تاني، اتصل بينا على طول. سلام!",
            'gulf': "مشكور! إذا تحتاج شي ثاني، اتصل علينا. سلامات!",
            'levantine': "شكراً إلك! إذا بدك شي تاني، اتصل فينا. سلام!",
            'moroccan': "شكراً ليك! إلا بغيتي شي آخر، عيط علينا. بسلامة!",
            'msa': "شكراً لك! إذا احتجت أي مساعدة أخرى، اتصل بنا. مع السلامة!"
        }
        return closings.get(dialect, closings['msa'])
