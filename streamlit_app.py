import os
import io
import base64
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Third-party imports
import streamlit as st
from dotenv import load_dotenv
import anthropic

# Global PDF reader
try:
    import pypdf
    PDF_READER_CLASS = pypdf.PdfReader
    logger.info("Successfully imported pypdf")
except ImportError:
    logger.warning("Failed to import pypdf, trying PyPDF2")
    try:
        import PyPDF2
        PDF_READER_CLASS = PyPDF2.PdfReader
        logger.info("Successfully imported PyPDF2")
    except ImportError as e:
        logger.error(f"Failed to import PDF libraries: {e}")
        st.error("PDF processing libraries not available. Please contact support.")
        PDF_READER_CLASS = None

# Load environment variables
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

def clean_extracted_text(text: str) -> str:
    """Clean and format extracted PDF text."""
    if not text:
        return ""
    text = text.replace('\n\n', '\n')
    text = text.strip()
    return text

def extract_pdf_content(pdf_path: str) -> Optional[str]:
    """Extract and process text content from a PDF file."""
    if not PDF_READER_CLASS:
        st.error("PDF processing is not available")
        return None

    logger.info(f"Starting PDF extraction from: {pdf_path}")
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return None

        with open(pdf_path, 'rb') as file:
            reader = PDF_READER_CLASS(file)
            text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
            return clean_extracted_text(" ".join(text))
    except Exception as e:
        logger.error(f"Error in PDF extraction: {str(e)}")
        return None

def get_reference_file(grade: str) -> Optional[str]:
    """Map grade levels to their reference PDF files."""
    grade_mapping = {
        "Kindergarten": "grade_3.pdf",
        "Grade 1": "grade_3.pdf",
        "Grade 2": "grade_3.pdf",
        "Grade 3": "grade_3.pdf",
        "Grade 4": "grade_4.pdf",
        "Grade 5": "grade_5.pdf",
        "Grade 6": "grade_6.pdf",
        "Grade 7": "grade_7.pdf",
        "Grade 8": "grade_8.pdf",
        "Algebra 1": "algebra_1.pdf",
        "Algebra 2": "algebra_1.pdf",
        "Geometry": "algebra_1.pdf"
    }
    return grade_mapping.get(grade)

def format_response(text: str) -> str:
    """Format the response with custom styling."""
    questions = text.split('Question')[1:]
    formatted_questions = []
    for q in questions:
        formatted_q = f'''
        <div style="
            background-color: #f8f9fa;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 4px solid #1f77b4;
        ">
            Question{q.replace(chr(10), '<br>')}
        </div>
        '''
        formatted_questions.append(formatted_q)
    return "".join(formatted_questions)

def get_response(grade: str, narrative: str, goals: str, standards: str, lessons: str) -> str:
    """Generate assessment content using the AI model."""
    client = anthropic.Anthropic(api_key=api_key)
    
    # Load reference material (used internally in the prompt only)
    reference_file = get_reference_file(grade)
    reference_text = ""
    if reference_file:
        pdf_path = f"reference_materials/{reference_file}"
        reference_text = extract_pdf_content(pdf_path) or ""
    
    # Construct prompt without displaying the reference material to users
    user_content = f"""
    # CONTEXT #
    You are creating a set of mathematics assessment items for {grade}.

    # Content Hierarchy (in order of priority):
    1. Narrative and Lesson Goals
    2. Goals
    3. Standards

    # Preliminary Steps:
    1. Review the Narrative, Lesson Goals, Goals, and Standards.
    2. Write a summary of how the Narrative and Lesson Goals drive the section’s key ideas
       and how they connect to the Goals and Standards. 
    3. From the Narrative and Lesson Goals, write a list of skills needed. 
       - Show how each skill helps learners meet the Goals and Standards.

    # Question Creation Guidelines:
    - Generate exactly 10 questions:
      - 5 Multiple Choice (MCQ) questions.
      - 5 Short Answer questions.
    - Ensure each question is solvable with the information provided.
    - Each question must reflect the Narrative and Lesson Goals first, then align with the Goals, and finally comply with the Standards.

    ## Multiple Choice Questions
    - Provide four answer options labeled A, B, C, D.
    - Each option should appear on its own line.
    
    ## Short Answer Questions
    - Require a clear step-by-step solution leading to a concise final answer.
    
    ## Visual Descriptions (if needed)
    - Place all visual descriptions in square brackets: [Visual: ...].
    - Include enough detail (coordinates, measures, etc.) so the problem is solvable.
    - Verify geometric or diagram-based details for mathematical consistency (e.g., parallel lines, angle sums, correct coordinates).
    
    # Formatting Requirements:
    1. Number each question as "Question 1:", "Question 2:", etc.
    2. Do your best to identify which of the submitted standards best aligns with the question and add this after the question number. Example, "Question 1: 8.8D"
    3. Use the following answer format for both MCQ and Short Answer:
       Answer: [Letter or numeric value] | Model Solution:
       • Step-by-step explanation
       • Final answer statement
    4. Do not include any meta-commentary or extra text beyond the 10 questions and their solutions.
    
    # Validation Checklist:
    1. Is the question solvable with the provided info?
    2. Does it reflect the Narrative and Lesson Goals first, then the Goals, then the Standards?
    3. Are visual or geometric details valid (correct angle sums, labeled measurements, etc.)?
    4. Is any diagram or measurement consistent and clearly labeled?
    5. Is there no missing or extraneous information?

    # REFERENCE FORMAT #
    Here are actual questions from the official {grade} assessment for content reference:

    {reference_text}

    # CONTENT TO ADDRESS #
    Generate questions covering:
    ***Learning Goals: {goals}
    ***Standards: {standards}
    ***Lesson Content: {lessons}
    ***Section Narrative: {narrative}

    Important: Generate all 10 questions at once. Do not include any introductory text, meta-commentary, or questions about continuing.
         
    3. Example Question Format for Multiple Choice:
       Question 1: 8.8D
       [Visual Description: Coordinate grid showing triangle ABC with vertices at (2,3), (4,8), and (6,2)]
       Triangle ABC has angle measures of 65° and 45°. What is the measure of the third angle?
       A) 60°
       B) 70°
       C) 85°
       D) 180°
       Answer: B | Model Solution:
       • Sum of angles in a triangle = 180°
       • Known angles: 65° + 45° = 110°
       • 180° - 110° = 70°
       Therefore, the third angle measures 70°

    """
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system="You are a mathematics assessment writer who exactly replicates official state assessment style and format.",
        messages=[
            {"role": "user", "content": user_content}
        ],
        max_tokens=4000,
        stream=False
    )
    
    return response.content[0].text

# Streamlit UI
st.title("Mathematics Assessment Generator")
st.subheader("Generate structured assessment items with rationales")
st.markdown("""
For samples and more information on how to collect the data for this tool, 
check out this [Help Page](https://docs.google.com/document/d/1S9gjx4meZiUfDb-b_W1Ca2yKjYirsjpZKpmbUJ5lMmw/edit?tab=t.0#heading=h.pjv2hotkik9a).
""")

# Grade Level Selection
grade = st.selectbox("Grade Level:", 
                     ["Kindergarten"] + [f"Grade {i}" for i in range(1, 9)] + 
                     ["Algebra 1", "Algebra 2", "Geometry"])

# Two-column layout for standards and lesson goals
col1, col2 = st.columns(2)
with col1:
    standards = st.text_area("Standards:", 
                             help="List the relevant content standards being addressed.",
                             height=150)
with col2:
    lessons = st.text_area("Lesson Learning Goals:", 
                           help="List the specific learning goals for each lesson in this section.",
                           height=150)

# Combined section narrative and learning goals
section_content = st.text_area("Section Narrative and Learning Goals:", 
                              help="Provide an overview of the content being covered in this section and the key learning goals.",
                              height=200)

# Generate response on button click
if st.button("Generate Assessment"):
    if all([grade, standards, lessons, section_content]):
        with st.spinner("Generating assessment..."):
            try:
                response_text = get_response(grade, section_content, lessons, standards, lessons)
                st.success("Assessment Generated Successfully!")
                
                # Display formatted output
                st.markdown(
                    format_response(response_text),
                    unsafe_allow_html=True
                )
                
                # Raw text for copying
                with st.expander("Show Raw Text"):
                    st.text_area("Raw Assessment Text", value=response_text, height=400)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error generating assessment: {str(e)}")
    else:
        st.warning("Please fill in all fields to generate the assessment.")
