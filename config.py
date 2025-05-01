import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
    TEMP_DIR = "temp"
    
    @classmethod
    def validate(cls):
        """Validate required environment variables are set"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Initialize configuration
config = Config()