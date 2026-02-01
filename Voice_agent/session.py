"""
Session Management for AI Call Center
Handles dialect locking and conversation history
"""

class CallSession:
    """
    Manages individual call session state
    """
    
    def __init__(self, session_id):
        self.session_id = session_id
        self.detected_dialect = None
        self.dialect_confidence = 0.0
        self.conversation_history = []
        self.dialect_locked = False
        self.lock_threshold = 0.8  # Confidence threshold for locking
    
    def lock_dialect(self, dialect, confidence):
        """
        Lock dialect once confidence threshold is met
        
        Args:
            dialect: Detected dialect (egyptian, gulf, levantine, moroccan, msa)
            confidence: Confidence score (0-1)
        
        Returns:
            bool: True if dialect was locked, False otherwise
        """
        if confidence >= self.lock_threshold and not self.dialect_locked:
            self.detected_dialect = dialect
            self.dialect_confidence = confidence
            self.dialect_locked = True
            print(f"✓ Dialect locked: {dialect} (confidence: {confidence:.2f})")
            return True
        
        # Update detected dialect even if not locked yet
        if confidence > self.dialect_confidence:
            self.detected_dialect = dialect
            self.dialect_confidence = confidence
        
        return False
    
    def add_interaction(self, user_text, assistant_text):
        """Store conversation turn"""
        self.conversation_history.append({
            'user': user_text,
            'assistant': assistant_text
        })
    
    def get_context(self, num_turns=3):
        """
        Get recent conversation context
        
        Args:
            num_turns: Number of recent turns to include
        
        Returns:
            list: Recent conversation history
        """
        return self.conversation_history[-num_turns:] if self.conversation_history else []
    
    def get_stats(self):
        """Get session statistics"""
        return {
            'session_id': self.session_id,
            'dialect': self.detected_dialect,
            'confidence': self.dialect_confidence,
            'locked': self.dialect_locked,
            'turns': len(self.conversation_history)
        }
