from langchain_groq import ChatGroq
from typing import List, Dict, Optional
import streamlit as st
import os
from dotenv import load_dotenv
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AnswerEvaluator:
    def __init__(self):
        """Initialize the AnswerEvaluator with Groq LLM"""
        logger.info("Initializing AnswerEvaluator")
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

    def _evaluate_answer_with_groq(self, prompt: str) -> str:
        """
        Evaluate the answer using Groq LLM synchronously
        """
        logger.debug(f"Evaluating answer with prompt length: {len(prompt)}")
        try:
            response = self.llm.invoke(prompt)
            logger.debug(f"Received response from Groq: {response.content[:100]}...")

            evaluation = response.content.strip()
            logger.info(f"Answer evaluation: {evaluation}")
            return evaluation

        except Exception as e:
            logger.error(f"Error in _evaluate_answer_with_groq: {str(e)}")
            st.error(f"Error evaluating answer with Groq: {str(e)}")
            return "Error in evaluation"

    def evaluate_answer(self, answer: str, question: str, cv_context: str, jd_context: str) -> Dict[str, Optional[str]]:
        """
        Evaluate the candidate's answer to a given interview question and return structured feedback.
        The feedback will include relevance, clarity, demonstration of skills, and alignment with job requirements.
        """
        logger.info("Evaluating answer")

        # Prepare prompt for answer evaluation
        prompt = f"""
You are an experienced interviewer. Please evaluate the following candidate's answer:

Question: {question}
Candidate's Answer: {answer}
Context: CV: {cv_context}, Job Description: {jd_context}

Provide feedback on the quality of the answer in terms of:
1. Relevance to the question
2. Clarity and structure
3. Demonstration of skills and experience
4. How well the answer aligns with the job requirements

Return a detailed evaluation in the form of a JSON object with the following keys:
- 'relevance' (string): Feedback on relevance to the question
- 'clarity' (string): Feedback on clarity and structure
- 'skills_demonstration' (string): Feedback on skills and experience demonstrated
- 'alignment' (string): Feedback on how well the answer aligns with the job requirements

For example:
{
  "relevance": "The answer is highly relevant to the question and covers all the key points.",
  "clarity": "The answer is clear, well-structured, and easy to understand.",
  "skills_demonstration": "The candidate demonstrated strong problem-solving skills and technical expertise.",
  "alignment": "The answer aligns well with the job requirements, showing a good understanding of the role."
}
        """

        # Get answer evaluation from Groq
        evaluation_json_str = self._evaluate_answer_with_groq(prompt)

        # Attempt to parse the JSON string to a dictionary
        try:
            evaluation_dict = json.loads(evaluation_json_str)
            logger.info(f"Parsed evaluation: {evaluation_dict}")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse the evaluation as JSON: {evaluation_json_str}")
            evaluation_dict = {
                'relevance': "Error parsing relevance feedback",
                'clarity': "Error parsing clarity feedback",
                'skills_demonstration': "Error parsing skills feedback",
                'alignment': "Error parsing alignment feedback"
            }

        return evaluation_dict
