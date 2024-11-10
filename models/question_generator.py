from langchain_groq import ChatGroq
from typing import List, Dict, Optional
from enum import Enum
import streamlit as st
import os
from dotenv import load_dotenv
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('question_generator.log')
    ]
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
                model_name="llama-3.2-90b-text-preview",
                temperature=0.7
            )
            logger.info("Successfully initialized Groq LLM")
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {str(e)}")
            raise

        # Define prompts for different question types
        self.type_prompts = {
            QuestionType.TECHNICAL: """
You are an experienced technical interviewer. Generate exactly 15 technical interview questions based on the following context:

Technical Skills Required: {technical_stack}
Candidate Background: {cv_context}
Job Requirements: {jd_context}

Rules for generating questions:
1. Questions should test practical coding and problem-solving skills
2. Include questions about system design and architecture
3. Focus on the specific technical stack mentioned
4. Match the complexity to the candidate's experience level
5. Each question should be clear and specific
6. Include questions about handling scalability or performance
7. Add questions about testing and code quality
8. Include questions about debugging and troubleshooting
9. Add questions about security best practices
10. Cover design patterns and architectural principles

Format the output exactly as follows:
1. First question here
2. Second question here
[...and so on until 15 questions]

Remember:
- Generate exactly 15 questions
- Number each question from 1 to 15
- Keep questions concise but specific
- Focus on practical scenarios
- Ensure questions are appropriate for the candidate's level""",

            QuestionType.BEHAVIORAL: """
You are an experienced HR interviewer. Generate exactly 15 behavioral interview questions based on the following context:

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
8. Include questions about diversity and inclusion
9. Cover remote work and collaboration scenarios
10. Add questions about mentoring and knowledge sharing

Format the output exactly as follows:
1. First question here
2. Second question here
[...and so on until 15 questions]

Remember:
- Generate exactly 15 questions
- Number each question from 1 to 15
- Start questions with "Tell me about a time when..." or similar phrases
- Focus on specific situations and experiences
- Ensure questions allow for STAR method responses""",

            QuestionType.COMPETENCY: """
You are an experienced competency-based interviewer. Generate exactly 15 competency-based questions using the following context:

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
8. Cover leadership and influence
9. Include questions about resource management
10. Address cross-functional collaboration

Format the output exactly as follows:
1. First question here
2. Second question here
[...and so on until 15 questions]

Remember:
- Generate exactly 15 questions
- Number each question from 1 to 15
- Focus on demonstrable competencies
- Ask for specific examples and scenarios
- Ensure questions reveal measurable outcomes"""
        }
        logger.info("Question templates initialized")

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

        # Format technical stack
        tech_stack_str = ", ".join(technical_stack) if technical_stack else "General technical skills"

        context = {
            "cv_context": cv_context,
            "jd_context": jd_context,
            "technical_stack": tech_stack_str
        }

        logger.debug(f"Prepared context: {context}")
        return context

    def _generate_with_groq(self, prompt: str) -> List[str]:
        """
        Generate questions using Groq LLM synchronously with improved error handling
        """
        logger.debug(f"Generating questions with prompt length: {len(prompt)}")
        try:
            # Add debug logging for the prompt
            logger.debug(f"Sending prompt to Groq: {prompt[:200]}...")

            # Convert the prompt to a message format
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages)

            # Log the raw response for debugging
            logger.debug(f"Raw Groq response: {response.content}")

            questions = []
            # Split response into lines and process each line
            for line in response.content.strip().split('\n'):
                line = line.strip()
                # Improved number detection regex
                if line and re.match(r'^\d{1,2}[\.\)]', line):
                    try:
                        # Extract question after the number and any delimiter
                        question = re.split(r'^\d{1,2}[\.\)]\s*', line)[1].strip()
                        questions.append(question)
                        logger.debug(f"Successfully extracted question: {question}")
                    except IndexError:
                        logger.warning(f"Failed to parse line: {line}")
                        continue

            logger.info(f"Generated {len(questions)} questions successfully")

            # If we got fewer than 15 questions, log a warning
            if len(questions) < 15:
                logger.warning(f"Only generated {len(questions)} questions, expected 15")

            # Ensure we always return exactly 15 questions
            while len(questions) < 15:
                default_q = f"Default question #{len(questions) + 1}"
                questions.append(default_q)
                logger.warning(f"Added default question: {default_q}")

            return questions[:15]

        except Exception as e:
            logger.error(f"Error in _generate_with_groq: {str(e)}", exc_info=True)
            st.error(f"Error generating questions with Groq: {str(e)}")
            # Return default questions on error
            default_questions = [f"Default question {i + 1}" for i in range(15)]
            return default_questions

    def generate_questions(self,
                           question_type: str,
                           resume_info: Optional[Dict] = None,
                           job_description: Optional[Dict] = None,
                           technical_stack: Optional[List[str]] = None) -> List[str]:
        """
        Generate interview questions based on type and context with improved error handling
        """
        logger.info(f"Generating questions of type: {question_type}")
        try:
            # Validate inputs
            if not question_type:
                raise ValueError("Question type is required")

            # Convert question type to match app's selection
            if question_type.lower() == "competency based":
                question_type = "competency_based"

            # Get question type enum
            try:
                q_type = QuestionType(question_type.lower())
            except ValueError as e:
                logger.error(f"Invalid question type: {question_type}")
                raise ValueError(
                    f"Invalid question type: {question_type}. Must be one of {[t.value for t in QuestionType]}")

            # Log input data
            logger.debug(f"Resume info: {resume_info}")
            logger.debug(f"Job description: {job_description}")
            logger.debug(f"Technical stack: {technical_stack}")

            # Prepare context
            context = self._prepare_context(resume_info, job_description, technical_stack)

            # Get prompt template and format it
            if q_type not in self.type_prompts:
                raise ValueError(f"No prompt template found for question type: {q_type}")

            prompt = self.type_prompts[q_type].format(**context)
            logger.debug(f"Generated prompt: {prompt[:200]}...")

            # Generate questions using Groq
            questions = self._generate_with_groq(prompt)

            # Validate output
            if not questions:
                logger.error("No questions generated")
                raise ValueError("Failed to generate questions")

            return questions

        except Exception as e:
            logger.error(f"Error in generate_questions: {str(e)}", exc_info=True)
            st.error(f"Error in question generation: {str(e)}")
            default_questions = [f"Default {question_type} question {i + 1}" for i in range(15)]
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