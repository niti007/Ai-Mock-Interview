
# AI Mock Interview Assistant

The AI Mock Interview Assistant is a Python application that provides personalized mock interviews based on a user's resume, job description, and the type of interview they select(technical, behavioral, and competency-based). Using natural language processing techniques, it analyzes the data and conducts mock interviews. Once the interview is finished, it gives you a summary of how well you answered the question and where you can improve.

## Features

- **Resume and Job Description Parsing**: Parses resumes (PDF or DOCX) and job descriptions to extract key details like skills, experience, education, and job requirements.
  
- **Interview Question generator**: Generate questions based on the type of interview selected and take youe cv and JD into context to generate question

- **Summary Analysis of the answers**: Creates tailored technical, behavioral, competency-based, and general interview questions derived from the resume and job description.

## Usage

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:

   ```bash
   python -m streamlit run app.py
   ```

3. **Follow On-Screen Instructions**: Upload your resume and job description, and provide interview performance data to receive customized recommendations and interview questions.

## Architecture

The AI Mock Interview Assistant comprises several key components:

- **CVParser**: Extracts relevant details from resumes, including skills, experience, and education.
  
- **JobDescriptionParser**: Extracts required skills, responsibilities, and qualifications from job descriptions.

- **ResourceRecommender**: Combines parsed data and interview performance to generate personalized resource recommendations.

- **QuestionGenerator**: Produces interview questions tailored to technical, behavioral, and competency-based interview types.

- **Answer Evaluator**: Evaluates your answer and provide you a detailed feedback.

- **Streamlit UI**: Provides an interactive and user-friendly interface.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvement, please submit a pull request or open an issue.


