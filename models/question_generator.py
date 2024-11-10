from langchain_groq import ChatGroq
from typing import List, Dict, Optional
from enum import Enum
import streamlit as st
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class QuestionType(Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    COMPETENCY = "competency_based"


class QuestionGenerator:
    def __init__(self):
        """Initialize the QuestionGenerator with Groq LLM"""
        logger.info("Initializing QuestionGenerator")
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            logger.error("Groq API key not found in environment variables")
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env file")

        try:
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name="llama-3.2-90b-text-preview"
            )
            logger.info("Successfully initialized Groq LLM")
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {str(e)}")
            raise

        # Define prompts for different question types
        self.type_prompts = {
            QuestionType.TECHNICAL: """
You are an experienced technical interviewer. Generate 10 technical interview questions based on the following context:

Technical Skills Required: {technical_stack}
Candidate Background: {cv_context}
Job Requirements: {jd_context}

Rules for generating questions:
1. Questions should test practical coding and problem-solving skills
2. Include questions about system design and architecture
3. Focus on the specific technical stack mentioned
4. Match the complexity to the candidate's experience level
5. Each question should be clear and specific
6. Include at least one question about handling scalability or performance
7. Add a question about testing and code quality

Please generate exactly 10 questions, formatted as a numbered list (1-10).
Just provide the questions without any additional text or explanations.""",

            QuestionType.BEHAVIORAL: """
You are an experienced HR interviewer. Generate 10 behavioral interview questions based on the following context:

Candidate Background: {cv_context}
Job Requirements: {jd_context}

Rules for generating questions:
1. Questions should follow the STAR format
2. Focus on past experiences and specific situations
3. Include questions about teamwork and leadership
4. Cover conflict resolution and problem-solving
5. Make questions relevant to the candidate's experience level
6. Include questions about handling pressure and deadlines
7. Add questions about adapting to change

Please generate exactly 10 questions, formatted as a numbered list (1-10).
Just provide the questions without any additional text or explanations.""",

            QuestionType.COMPETENCY: """
You are an experienced competency-based interviewer. Generate 10 competency-based questions using the following context:

Candidate Background: {cv_context}
Job Requirements: {jd_context}

Rules for generating questions:
1. Focus on specific skills and competencies required for the role
2. Include questions about project management and delivery
3. Cover communication and stakeholder management
4. Address decision-making and problem-solving abilities
5. Make questions measurable and evidence-based
6. Include questions about innovation and continuous improvement
7. Add questions about strategic thinking and planning

Please generate exactly 10 questions, formatted as a numbered list (1-10).
Just provide the questions without any additional text or explanations."""
        }
        logger.info("Question templates initialized")

    def _generate_with_groq(self, prompt: str) -> List[str]:
        """
        Generate questions using Groq LLM synchronously
        """
        logger.debug(f"Generating questions with prompt length: {len(prompt)}")
        try:
            response = self.llm.invoke(prompt)
            logger.debug(f"Received response from Groq: {response.content[:100]}...")

            questions = []
            for line in response.content.strip().split('\n'):
                line = line.strip()
                if line and any(line.startswith(str(i)) for i in range(1, 6)):
                    question = line.split('.', 1)[1].strip() if '.' in line else line
                    questions.append(question)
                    logger.debug(f"Extracted question: {question}")

            logger.info(f"Generated {len(questions)} questions successfully")
            return questions[:5]

        except Exception as e:
            logger.error(f"Error in _generate_with_groq: {str(e)}")
            st.error(f"Error generating questions with Groq: {str(e)}")
            return []

    def _prepare_context(self, resume_info: Optional[Dict],
                         job_description: Optional[Dict],
                         technical_stack: Optional[List[str]] = None) -> Dict:
        """
        Prepare context for question generation
        """
        logger.debug("Preparing context for question generation")
        logger.debug(f"Resume info: {resume_info}")
        logger.debug(f"Job description: {job_description}")
        logger.debug(f"Technical stack: {technical_stack}")

        # Format CV context
        cv_context = "Not provided"
        if resume_info:
            cv_parts = []
            if 'skills' in resume_info:
                cv_parts.append(f"Skills: {', '.join(resume_info['skills'])}")
            if 'experience' in resume_info:
                cv_parts.append(f"Experience: {resume_info['experience']}")
            if 'education' in resume_info:
                cv_parts.append(f"Education: {resume_info['education']}")
            cv_context = ". ".join(cv_parts)

        # Format JD context
        jd_context = "Not provided"
        if job_description:
            jd_parts = []
            if 'requirements' in job_description:
                jd_parts.append(f"Requirements: {', '.join(job_description['requirements'])}")
            if 'responsibilities' in job_description:
                jd_parts.append(f"Responsibilities: {', '.join(job_description['responsibilities'])}")
            jd_context = ". ".join(jd_parts)

        context = {
            "cv_context": cv_context,
            "jd_context": jd_context,
            "technical_stack": ", ".join(technical_stack) if technical_stack else "General technical skills"
        }

        logger.debug(f"Prepared context: {context}")
        return context

    def generate_questions(self,
                           question_type: str,
                           resume_info: Optional[Dict] = None,
                           job_description: Optional[Dict] = None,
                           technical_stack: Optional[List[str]] = None) -> List[str]:
        """
        Generate interview questions based on type and context (synchronous version)
        """
        logger.info(f"Generating questions of type: {question_type}")
        try:
            # Convert question type to match app's selection
            if question_type.lower() == "competency based":
                question_type = "competency_based"

            # Get question type enum
            q_type = QuestionType(question_type.lower())
            logger.debug(f"Mapped question type to enum: {q_type}")

            # Prepare context
            context = self._prepare_context(resume_info, job_description, technical_stack)

            # Get prompt template and format it
            prompt = self.type_prompts[q_type].format(**context)
            logger.debug(f"Generated prompt of length: {len(prompt)}")

            # Generate questions using Groq
            questions = self._generate_with_groq(prompt)
            logger.info(f"Generated {len(questions)} questions")

            # If we didn't get enough questions, add some defaults
            while len(questions) < 5:
                default_q = f"Default {question_type} question #{len(questions) + 1}"
                questions.append(default_q)
                logger.warning(f"Added default question: {default_q}")

            return questions

        except Exception as e:
            logger.error(f"Error in generate_questions: {str(e)}", exc_info=True)
            st.error(f"Error in question generation: {str(e)}")
            default_questions = [f"Default {question_type} question {i + 1}" for i in range(5)]
            logger.info("Returning default questions due to error")
            return default_questions


def test_question_generator():
    """
    Test function for QuestionGenerator
    """
    try:
        logger.info("Starting QuestionGenerator test")

        # Test data
        test_resume = {
            "skills": ["Python", "Machine Learning", "API Development"],
            "experience": "3 years as Software Engineer, 2 years as ML Engineer",
            "education": "MS in Computer Science"
        }

        test_job = {
            "requirements": ["Python expertise", "ML knowledge", "API design"],
            "responsibilities": ["Lead ML projects", "Design APIs"],
            "role": "Senior Software Engineer"
        }

        # Initialize generator
        generator = QuestionGenerator()

        # Test each question type
        for q_type in QuestionType:
            logger.info(f"Testing question type: {q_type.value}")
            questions = generator.generate_questions(
                question_type=q_type.value,
                resume_info=test_resume,
                job_description=test_job,
                technical_stack=["Python", "Machine Learning"] if q_type == QuestionType.TECHNICAL else None
            )

            logger.info(f"Generated {len(questions)} questions for {q_type.value}")
            for i, q in enumerate(questions, 1):
                logger.debug(f"Question {i}: {q}")

        logger.info("QuestionGenerator test completed successfully")

    except Exception as e:
        logger.error(f"Error in test_question_generator: {str(e)}", exc_info=True)


if __name__ == "__main__":
    test_question_generator()