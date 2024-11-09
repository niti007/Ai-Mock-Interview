from langchain_groq import ChatGroq
from typing import List, Dict, Optional
from enum import Enum
import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QuestionType(Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    COMPETENCY = "competency_based"
    GENERAL = "general"

class QuestionGenerator:
    def __init__(self):
        """Initialize the QuestionGenerator with Groq LLM"""
        # Get API key from environment variable
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env file")

        # Initialize Groq LLM
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name="llama-3.2-90b-text-preview"
        )

        # Define prompts for different question types
        self.type_prompts = {
            QuestionType.TECHNICAL: """
You are an experienced technical interviewer. Generate 5 technical interview questions based on the following context:

Technical Skills Required: {technical_stack}
Candidate Background: {cv_context}
Job Requirements: {jd_context}

Rules for generating questions:
1. Questions should test practical coding and problem-solving skills
2. Include questions about system design and architecture
3. Focus on the specific technical stack mentioned
4. Match the complexity to the candidate's experience level
5. Each question should be clear and specific

Please generate exactly 5 questions, formatted as a numbered list (1-5).
Just provide the questions without any additional text or explanations.""",

            QuestionType.BEHAVIORAL: """
You are an experienced HR interviewer. Generate 5 behavioral interview questions based on the following context:

Candidate Background: {cv_context}
Job Requirements: {jd_context}

Rules for generating questions:
1. Questions should follow the STAR format
2. Focus on past experiences and specific situations
3. Include questions about teamwork and leadership
4. Cover conflict resolution and problem-solving
5. Make questions relevant to the candidate's experience level

Please generate exactly 5 questions, formatted as a numbered list (1-5).
Just provide the questions without any additional text or explanations.""",

            QuestionType.COMPETENCY: """
You are an experienced competency-based interviewer. Generate 5 competency-based questions using the following context:

Candidate Background: {cv_context}
Job Requirements: {jd_context}

Rules for generating questions:
1. Focus on specific skills and competencies required for the role
2. Include questions about project management and delivery
3. Cover communication and stakeholder management
4. Address decision-making and problem-solving abilities
5. Make questions measurable and evidence-based

Please generate exactly 5 questions, formatted as a numbered list (1-5).
Just provide the questions without any additional text or explanations.""",

            QuestionType.GENERAL: """
You are an experienced interviewer. Generate 5 general interview questions based on the following context:

Candidate Background: {cv_context}
Job Requirements: {jd_context}

Rules for generating questions:
1. Include a mix of professional and personal questions
2. Focus on understanding the candidate's motivations and goals
3. Cover work style and preferences
4. Include questions about career aspirations
5. Keep questions open-ended but specific

Please generate exactly 5 questions, formatted as a numbered list (1-5).
Just provide the questions without any additional text or explanations."""
        }

    async def _generate_with_groq(self, prompt: str) -> List[str]:
        """
        Generate questions using Groq LLM

        Args:
            prompt (str): The formatted prompt to send to Groq

        Returns:
            List[str]: List of generated questions
        """
        try:
            # Use asyncio to run the synchronous Groq call in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm.invoke(prompt)
            )

            # Extract questions from response
            questions = []
            for line in response.content.strip().split('\n'):
                line = line.strip()
                if line and any(line.startswith(str(i)) for i in range(1, 6)):
                    question = line.split('.', 1)[1].strip() if '.' in line else line
                    questions.append(question)

            return questions[:5]  # Ensure we only return 5 questions

        except Exception as e:
            st.error(f"Error generating questions with Groq: {str(e)}")
            return []

    def _prepare_context(self, resume_info: Optional[Dict],
                         job_description: Optional[Dict],
                         technical_stack: Optional[List[str]] = None) -> Dict:
        """
        Prepare context for question generation

        Args:
            resume_info (Dict): Parsed resume information
            job_description (Dict): Parsed job description
            technical_stack (List[str]): List of technical skills for technical interviews

        Returns:
            Dict: Formatted context for prompt generation
        """
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

        return {
            "cv_context": cv_context,
            "jd_context": jd_context,
            "technical_stack": ", ".join(technical_stack) if technical_stack else "General technical skills"
        }

    async def generate_questions(self,
                                 question_type: str,
                                 resume_info: Optional[Dict] = None,
                                 job_description: Optional[Dict] = None,
                                 technical_stack: Optional[List[str]] = None) -> List[str]:
        """
        Generate interview questions based on type and context.

        Args:
            question_type (str): Type of questions to generate
            resume_info (Dict, optional): Parsed resume information
            job_description (Dict, optional): Parsed job description
            technical_stack (List[str], optional): List of technical skills for technical interviews

        Returns:
            List[str]: List of generated questions
        """
        try:
            # Convert question type to match app's selection
            if question_type.lower() == "competency based":
                question_type = "competency_based"

            # Get question type enum
            q_type = QuestionType(question_type.lower())

            # Prepare context
            context = self._prepare_context(resume_info, job_description, technical_stack)

            # Get prompt template and format it
            prompt = self.type_prompts[q_type].format(**context)

            # Generate questions using Groq
            questions = await self._generate_with_groq(prompt)

            # If we didn't get enough questions, add some defaults
            while len(questions) < 5:
                questions.append(f"Default {question_type} question #{len(questions) + 1}")

            return questions

        except Exception as e:
            st.error(f"Error in question generation: {str(e)}")
            return [f"Default {question_type} question {i + 1}" for i in range(5)]


# Example usage and testing
if __name__ == "__main__":
    async def test_generator():
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

        # Test technical questions
        print("\nTechnical Questions:")
        technical_questions = await generator.generate_questions(
            question_type="technical",
            resume_info=test_resume,
            job_description=test_job,
            technical_stack=["Python", "Machine Learning"]
        )
        for i, q in enumerate(technical_questions, 1):
            print(f"{i}. {q}")

    # Run the test
    asyncio.run(test_generator())