from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI

# Load environment variables
load_dotenv()

class ChatbotManager:
    def __init__(self):
        # Initialize the Gemini LLM model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,  # Lower temperature for more factual responses
            max_tokens=1024
        )
        
        # Initialize Sutra API client for translations
        self.sutra_client = OpenAI(
            base_url="https://api.two.ai/v2",
            api_key=os.getenv("SUTRA_API_KEY")
        )
        
        # Define engineering education prompt template
        self.system_prompt = """###Do not respond to inappropriate, irrelevant queries which do not relate to Engineering or Science domains.
        You are LearnEasy, an advanced educational assistant specializing in Engineering subjects for students who are pursuinhg their Bachelors or Masters in Engineering.
Your primary purpose is to help engineering students understand complex technical concepts, solve problems, and improve their academic performance. 

**EXPERTISE AREAS:**
- Computer Science & IT: Programming, Data Structures and Algorithms(DSA), Database Management Systems, Operating Systems, Computer Networks, Software Engineering, SDLC, Web Development, Cloud Computing, Cybersecurity, Computer Architecture, AI & ML, Data Science
- Electrical Engineering: Circuit Theory, Signals & Systems, Digital Electronics, Power Systems, Control Systems, Microprocessors
- Mechanical Engineering: Thermodynamics, Fluid Mechanics, Machine Design, Manufacturing Processes, Heat Transfer, Engineering Mechanics
- Electronics & Communication: Analog & Digital Communications, Electromagnetic Theory, VLSI Design, Embedded Systems
- Core Engineering Subjects: Engineering Mathematics, Engineering Physics, Engineering Graphics, Engineering Chemistry
- Humanities and Social Sciences: Engineering Ethics, Professional Communication, Management Principles

GUIDELINES:
1. Focus exclusively on educational content related to engineering disciplines.
2. Provide clear, accurate explanations with relevant examples.
3. For problem-solving queries, show step-by-step solutions with explanations.
4. Include diagrams, equations, or code snippets when appropriate (using markdown).
5. Cite authoritative engineering resources when possible.
6. Respond in a helpful, encouraging manner suitable for students.
7. Do not provide answers to questions that appear to be active homework or exams.
8. Refuse to answer non-engineering related questions and gently redirect to engineering topics.

When responding to complex topics, structure your answers with clear headings, bullet points, and a logical flow to enhance understanding.
"""
        
        # Create prompt template with system message and conversation history
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
        
        # Create an LLM chain with the prompt template
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )
        
        # Store user sessions
        # session_id -> {messages: List, last_active: datetime}
        self.sessions: Dict[str, Dict] = {}
        
    def get_or_create_session(self, session_id: str) -> Dict:
        """Get existing session or create a new one if it doesn't exist"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "messages": [],
                "last_active": datetime.now()
            }
        else:
            # Update last active timestamp
            self.sessions[session_id]["last_active"] = datetime.now()
            
        return self.sessions[session_id]
    
    async def process_query(self, session_id: str, query: str, language: str = "english") -> Dict:
        """
        Process a user query and return the response in the specified language.
        Language options: english (default), bengali, hindi
        """
        session = self.get_or_create_session(session_id)
        
        # Add the user message to the session history
        user_message = HumanMessage(content=query)
        session["messages"].append(user_message)
        
        try:
            # Extract history in format suitable for prompt template
            history = session["messages"][:-1]  # All messages except the latest user query
            
            # Get response from the LLM using the prompt template
            response_text = await self.chain.ainvoke({
                "history": history,
                "query": query
            })
            
            ai_response = response_text["text"]
            
            # Handle translation if required
            language = language.lower()
            if language in ["bengali", "hindi"]:
                translated_response = await self.translate_to_language(ai_response, language)
                ai_message = AIMessage(content=translated_response)
            else:
                # Default English response
                ai_message = AIMessage(content=ai_response)
            
            # Add the AI response to the session history
            session["messages"].append(ai_message)
            
            return {
                "session_id": session_id,
                "response": ai_message.content,
                "created_at": datetime.now().isoformat(),
                "success": True
            }
            
        except Exception as e:
            return {
                "session_id": session_id,
                "response": f"Error processing your request: {str(e)}",
                "created_at": datetime.now().isoformat(),
                "success": False
            }

    async def translate_to_language(self, text: str, target_language: str) -> str:
        """
        Translate the given text to the target language using Sutra API
        """
        try:
            # Create a translation prompt based on the target language
            translation_prompt = f"Translate the following English text to {target_language}:\n\n{text}"
            
            # Call Sutra API for translation
            stream = self.sutra_client.chat.completions.create(
                model="sutra-v2",
                messages=[{"role": "user", "content": translation_prompt}],
                max_tokens=2048,  # Increased to accommodate longer translations
                temperature=0,
                stream=True
            )

            translated_text = ""
            for chunk in stream:
                if len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    finish_reason = chunk.choices[0].finish_reason
                    if content and finish_reason is None:
                        translated_text += content
            
            return translated_text
        
        except Exception as e:
            # If translation fails, return original text with error note
            return f"{text}\n\n[Translation Error: {str(e)}]"

    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        session = self.get_or_create_session(session_id)
        history = []
        
        for message in session["messages"]:
            if isinstance(message, HumanMessage):
                history.append({
                    "role": "user",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                history.append({
                    "role": "assistant",
                    "content": message.content
                })
        
        return history
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions that haven't been active for a specified time"""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session_data in self.sessions.items():
            time_diff = current_time - session_data["last_active"]
            if time_diff.total_seconds() > max_age_hours * 3600:
                sessions_to_remove.append(session_id)
                
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            
        return len(sessions_to_remove)