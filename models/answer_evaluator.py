from langchain_groq import ChatGroq
from typing import Dict, Optional
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
You are an experienced interviewer tasked with evaluating a candidate's answer.

Question: {question}
Candidate's Answer: {answer}
Context:
- CV Summary: {cv_context}
- Job Description: {jd_context}

Please provide detailed feedback in the following areas:
1. **Relevance**: Does the answer directly address the question and highlight necessary skills?
2. **Clarity and Structure**: Is the answer coherent, logical, and easy to understand?
3. **Skills Demonstration**: How well does the answer showcase relevant skills, knowledge, and problem-solving abilities?
4. **Alignment with Job Requirements**: Does the answer meet the specific requirements and expectations of the job?

Return feedback in JSON format with the keys:
- 'relevance'
- 'clarity'
- 'skills_demonstration'
- 'alignment'

For example:
{{
  "relevance": "The answer is highly relevant and addresses all key points in detail.",
  "clarity": "The response is clear, organized, and well-articulated.",
  "skills_demonstration": "The candidate effectively demonstrates relevant technical skills and industry knowledge.",
  "alignment": "The answer aligns well with job requirements, particularly in areas of technical expertise and role-specific knowledge."
}}
        """

        # Get answer evaluation from Groq
        evaluation_json_str = self._evaluate_answer_with_groq(prompt)

        # Attempt to parse the JSON string to a dictionary
        try:
            evaluation_dict = json.loads(evaluation_json_str)
            if all(key in evaluation_dict for key in ["relevance", "clarity", "skills_demonstration", "alignment"]):
                logger.info(f"Parsed evaluation: {evaluation_dict}")
            else:
                logger.warning("Some expected keys are missing in the response. Returning defaults.")
                evaluation_dict = {
                    'relevance': evaluation_dict.get('relevance', "Incomplete relevance feedback"),
                    'clarity': evaluation_dict.get('clarity', "Incomplete clarity feedback"),
                    'skills_demonstration': evaluation_dict.get('skills_demonstration', "Incomplete skills feedback"),
                    'alignment': evaluation_dict.get('alignment', "Incomplete alignment feedback")
                }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse the evaluation as JSON: {evaluation_json_str}")
            evaluation_dict = {
                'relevance': "Error parsing relevance feedback",
                'clarity': "Error parsing clarity feedback",
                'skills_demonstration': "Error parsing skills feedback",
                'alignment': "Error parsing alignment feedback"
            }

        return evaluation_dict
