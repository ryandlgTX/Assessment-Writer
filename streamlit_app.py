import os
import base64
from dotenv import load_dotenv
import anthropic
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

# Verify API key
if not api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()

def get_reference_file(grade):
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

def load_pdf_by_grade(grade):
    """Load PDF reference material based on selected grade level."""
    try:
        reference_file = get_reference_file(grade)
        if not reference_file:
            st.warning(f"No reference material mapping found for {grade}")
            return None
            
        pdf_path = f"reference_materials/{reference_file}"
        with open(pdf_path, "rb") as file:
            return base64.b64encode(file.read()).decode('utf-8')
    except Exception as e:
        st.warning(f"Error loading reference material for {grade}: {str(e)}")
        return None

def get_response(grade, narrative, goals, standards, lessons):
    """Send user input to the AI model and get a response using the Messages API."""
    client = anthropic.Anthropic(api_key=api_key)
    
    # Constructing the refined prompt
    user_content = f"""
    # CONTEXT #
    You are a mathematics assessment expert. Generate exactly 10 questions (5 multiple choice and 5 short answer) based on the provided content. Complete all analysis internally - do not show your work. Return only the formatted questions, rationales, and model responses.

    # OBJECTIVE #
    >>>INPUTS
    ***Grade Level: {grade}
    ***Section Narrative: {narrative}
    ***Section Learning Goals: {goals}
    ***Standards: {standards}
    ***Lesson Learning Goals: {lessons}

    Follow these steps to generate the assessment content:

    >>>STEP 1 - Build your knowledge base
    a. Review the narrative, goals, lessons, and standards to understand the content. Write a summary of your understanding.
    b. Create a list of skills needed for success in this section.
    c. For each skill, write a description of how it connects to the section content.

    >>>STEP 2 - Develop Questions
    a. Draft 10 questions (5 multiple choice and 5 short answer) addressing the identified skills.
    b. Review questions against lesson goals and standards to ensure appropriate coverage.
    c. Write a rationale for each question connecting it to specific skills and content.

    # ADDITIONAL CONSIDERATIONS #
    >>>Structure Questions as Progressive Sequences
    - Start with direct computation
    - Build to pattern recognition
    - End with application and explanation
    - Keep context consistent across parts

    >>>Context Guidelines
    - Use rich, realistic situations that naturally connect to unit content
    - Maintain same context across multiple parts
    - Ensure numbers and situations are grade-appropriate

    >>>Required Question Components
    - Direct skill practice
    - Pattern recognition/analysis
    - Written explanation/justification
    - Creation of examples or application to new situations

    >>>Language Requirements
    - Use consistent mathematical vocabulary
    - Include specific prompts: "explain why," "show another way," "create an example"
    - Frame questions to elicit complete mathematical thoughts

    # OUTPUT FORMAT #
    Return ONLY the questions, rationales, and model responses in this exact format, with no additional text:

    Question 1: [Question text]
    A) [Option]
    B) [Option]
    C) [Option]
    D) [Option]
    Rationale: [Explanation connecting to skills and content]
    Model Response: [Letter] | [Complete student work showing understanding]

    Question 2: [Short answer question text]
    Rationale: [Explanation connecting to skills and content]
    Model Response: [Complete student work showing understanding]

    [Continue through Question 10]

    Do not include any preliminary analysis, introductory text, or questions about continuing. Generate all 10 questions at once.
    """
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system="You are a mathematics assessment expert that creates grade-appropriate practice questions with clear rationales.",
        messages=[
            {"role": "user", "content": user_content}
        ],
        max_tokens=4000,
        stream=False
    )
    
    return response.content[0].text

# Streamlit app UI
st.title("Mathematics Assessment Generator")
st.subheader("Generate structured practice questions with rationales")

# Input fields
grade_options = ["Kindergarten"] + [f"Grade {i}" for i in range(1, 9)] + ["Algebra 1", "Algebra 2", "Geometry"]
grade = st.selectbox("Grade Level:", grade_options)
narrative = st.text_area("Section Narrative:", 
                        help="Provide an overview of the content being covered in this section.",
                        height=150)
goals = st.text_area("Section Learning Goals:", 
                     help="List the key learning goals for this section.",
                     height=100)
standards = st.text_area("Standards:", 
                        help="List the relevant content standards being addressed.",
                        height=100)
lessons = st.text_area("Lesson Learning Goals:", 
                      help="List the specific learning goals for each lesson in this section.",
                      height=150)

# Function to create formatted HTML for the response
def format_response(response):
    # Split response into questions
    questions = response.split('\n\n')
    formatted_html = ""
    
    for question in questions:
        if not question.strip():
            continue
        
        # Add custom styling for each question block
        formatted_html += (
            '<div style="'
            'background-color: #f8f9fa;'
            'padding: 20px;'
            'margin: 10px 0;'
            'border-radius: 5px;'
            'border-left: 4px solid #1f77b4;'
            '">'
            f'{question.replace(chr(10), "<br>")}'
            '</div>'
        )
    
    return formatted_html

# Generate response
if st.button("Generate Assessment"):
    if all([grade, narrative, goals, standards, lessons]):
        with st.spinner("Generating assessment questions and rationales..."):
            try:
                response = get_response(grade, narrative, goals, standards, lessons)
                st.success("Assessment Generated Successfully!")
                
                # Display formatted output
                st.markdown(
                    format_response(response),
                    unsafe_allow_html=True
                )
                
                # Provide raw text area for easy copying
                with st.expander("Show Raw Text"):
                    st.text_area("Raw Assessment Text", value=response, height=400)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please fill in all fields to generate the assessment.")