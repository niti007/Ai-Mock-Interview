
# Resource Recommender

The **Resource Recommender** is a Python application that provides personalized learning resource recommendations based on a user's resume, job description, and interview performance. Using natural language processing techniques, it analyzes the provided data to suggest targeted resources for technical, behavioral, and competency-based interview preparation.

## Features

- **Resume and Job Description Parsing**: Parses resumes (PDF or DOCX) and job descriptions to extract key details like skills, experience, education, and job requirements.
  
- **Interview Performance Analysis**: Analyzes interview performance, identifies areas of improvement, and recommends resources to strengthen weak areas.

- **Personalized Resource Recommendations**: Generates resource suggestions prioritized by skill development, interview preparation, and additional resources based on parsed data and performance analysis.

- **Flexible Question Generation**: Creates tailored technical, behavioral, competency-based, and general interview questions derived from the resume and job description.

- **Audio Transcription**: Records audio from the user's microphone and transcribes responses using the Whisper speech recognition model.

## Usage

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:

   ```bash
   python app.py
   ```

3. **Follow On-Screen Instructions**: Upload your resume and job description, and provide interview performance data to receive customized recommendations and interview questions.

## Architecture

The Resource Recommender comprises several key components:

- **CVParser**: Extracts relevant details from resumes, including skills, experience, and education.
  
- **JobDescriptionParser**: Extracts required skills, responsibilities, and qualifications from job descriptions.

- **ResourceRecommender**: Combines parsed data and interview performance to generate personalized resource recommendations.

- **QuestionGenerator**: Produces interview questions tailored to technical, behavioral, and competency-based interview types.

- **AudioHandler**: Handles audio input and transcription with the Whisper model.

- **Streamlit UI**: Provides an interactive and user-friendly interface.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvement, please submit a pull request or open an issue.


