import streamlit as st
from generator import get_response, chat_message
import time
 
# background
st.image('./assets/streamlit background  UniGuid.png')

# intro
st.title("Welcome to UniGuide App 🎓️")
st.markdown('*We are here to help you ❤️*')
st.text('UniGuide is a smart, AI-powered app designed to make university life easier for students. Whether you need help with course registration, academic advice, campus resources, or time management, UniGuide has got you covered. With personalized recommendations, real-time assistance, and an intuitive interface, the app ensures you stay on top of your studies while making the most of your university experience. Say goodbye to confusion and hello to a smoother academic journey with UniGuide! 🚀')
st.markdown('---')

# course selections
st.header('Please select your completed couses')

# Level 1
semester_1 = [
    "Pharmaceutical Analytical Chemistry I",
    "Pharmaceutical Organic Chemistry I",
    "Pharmacy Orientation",
    "English",
    "Medicinal Plants",
    "Information Technology",
    "Mathematics",
    "Human Rights and Fighting Corruption"
]

semester_2 = [
    "Pharmaceutical Analytical Chemistry II",
    "Pharmaceutical Organic Chemistry II",
    "Cell Biology",
    "Medical Terminology",
    "Anatomy & Histology",
    "Physical Pharmacy",
    "Pharmacognosy I",
    "Psychology"
]

# Level 2
semester_3 = [
    "Pharmaceutical Analytical Chemistry III",
    "Pharmaceutical Organic Chemistry III",
    "Pharmacognosy II",
    "Physiology and Pathophysiology",
    "Pharmaceutics I",
    "Scientific Writing",
    "Scientific Thinking",
    "Sinai History"
]

semester_4 = [
    "Biochemistry I",
    "General Microbiology and Immunology",
    "Instrumental Analysis",
    "Pathology",
    "Pharmaceutics II",
    "Presentation & Communication Skills",
    "Biostatistics"
]

# Level 3
semester_5 = [
    "Biochemistry II",
    "Pharmaceutical Microbiology",
    "Phytochemistry",
    "Pharmaceutics III",
    "Medicinal Chemistry I",
    "Pharmacology I"
]

semester_6 = [
    "Parasitology",
    "Biopharmaceutics & Pharmacokinetics",
    "Applied and Forensic Pharmacognosy",
    "Pharmaceutics IV",
    "Pharmacology II",
    "Medicinal Chemistry II"
]

# Level 4
semester_7 = [
    "Medical Microbiology",
    "Pharmacology III",
    "Drug Design",
    "Clinical Biochemistry",
    "Pharmaceutical Technology I",
    "Pharmaceutical Legislations & Professional Ethics",
    "Elective"
]

semester_8 = [
    "Clinical Pharmacokinetics",
    "Drug Information",
    "Toxicology & Forensic Chemistry",
    "Hospital Pharmacy",
    "Pharmaceutical Technology II",
    "Clinical Pharmacy and Pharmacotherapeutics I",
    "Elective"
]

# Level 5
semester_9 = [
    "Biotechnology",
    "Community Pharmacy Practice",
    "Public Health",
    "Phytotherapy and Aromatherapy",
    "Good Manufacturing Practice",
    "Marketing & Pharmacoeconomics",
    "Clinical Pharmacy and Pharmacotherapeutics II",
    "Elective"
]

semester_10 = [
    "Quality Control of Pharmaceuticals",
    "First Aid",
    "Drug Interaction",
    "Advanced Drug Delivery Systems",
    "Clinical Pharmacy and Pharmacotherapeutics III",
    "Entrepreneurship",
    "Clinical Research, Pharmacoepidemiology & Pharmacovigilance",
    "Elective"
]

st.subheader('Level One: Semester One')
semester_1_selected = st.pills('s1', options= semester_1, selection_mode="multi", label_visibility='collapsed')

st.subheader('Level One: Semester Two')
semester_2_selected = st.pills('s2', options= semester_2, selection_mode="multi", label_visibility='collapsed')

st.subheader('Level Two: Semester One')
semester_3_selected = st.pills('s3', options= semester_3, selection_mode="multi", label_visibility='collapsed')

st.subheader('Level Two: Semester Two')
semester_4_selected = st.pills('s4', options= semester_4, selection_mode="multi", label_visibility='collapsed')

st.subheader('Level Three: Semester One')
semester_5_selected = st.pills('s5', options= semester_5, selection_mode="multi", label_visibility='collapsed')

st.subheader('Level Three: Semester Two')
semester_6_selected = st.pills('s6', options= semester_6, selection_mode="multi", label_visibility='collapsed')

st.subheader('Level Four: Semester One')
semester_7_selected = st.pills('s7', options= semester_7, selection_mode="multi", label_visibility='collapsed')

st.subheader('Level Four: Semester Two')
semester_8_selected = st.pills('s8', options= semester_8, selection_mode="multi", label_visibility='collapsed')

st.subheader('Level Five: Semester One')
semester_9_selected = st.pills('s9', options= semester_9, selection_mode="multi", label_visibility='collapsed')

st.subheader('Level Five: Semester Two')
semester_10_selected = st.pills('s10', options= semester_10, selection_mode="multi", label_visibility='collapsed')

student_selected_courses = semester_1_selected + semester_2_selected + semester_3_selected + semester_4_selected + semester_5_selected + semester_6_selected + semester_7_selected + semester_8_selected + semester_9_selected + semester_10_selected


# prompt
pharmacy_courses = {
    "Level 1": {
        "Semester 1": {
            1: {"Course Code": "PC 101", "Course Name": "Pharmaceutical Analytical Chemistry I", "Credit Hours": 3, "Prerequisites": "Registration"},
            2: {"Course Code": "PC 102", "Course Name": "Pharmaceutical Organic Chemistry I", "Credit Hours": 3, "Prerequisites": "Registration"},
            3: {"Course Code": "PT 101", "Course Name": "Pharmacy Orientation", "Credit Hours": 1, "Prerequisites": "Registration"},
            4: {"Course Code": "LNG1001", "Course Name": "English", "Credit Hours": 3, "Prerequisites": "Registration"},
            5: {"Course Code": "PG 101", "Course Name": "Medicinal Plants", "Credit Hours": 3, "Prerequisites": "Registration"},
            6: {"Course Code": "NP 101", "Course Name": "Information Technology", "Credit Hours": 3, "Prerequisites": "Registration"},
            7: {"Course Code": "MS 101", "Course Name": "Mathematics", "Credit Hours": 1, "Prerequisites": "Registration"},
            8: {"Course Code": "NP 102", "Course Name": "Human Rights and Fighting Corruption", "Credit Hours": 1, "Prerequisites": "Registration"}
        },
        "Semester 2": {
            1: {"Course Code": "PC 203", "Course Name": "Pharmaceutical Analytical Chemistry II", "Credit Hours": 3, "Prerequisites": "Pharmaceutical Analytical Chemistry I"},
            2: {"Course Code": "PC 204", "Course Name": "Pharmaceutical Organic Chemistry II", "Credit Hours": 3, "Prerequisites": "Pharmaceutical Organic Chemistry I"},
            3: {"Course Code": "MD 201", "Course Name": "Cell Biology", "Credit Hours": 2, "Prerequisites": "Registration"},
            4: {"Course Code": "MD 202", "Course Name": "Medical Terminology", "Credit Hours": 1, "Prerequisites": "Registration"},
            5: {"Course Code": "MD 203", "Course Name": "Anatomy & Histology", "Credit Hours": 3, "Prerequisites": "Registration"},
            6: {"Course Code": "PT 202", "Course Name": "Physical Pharmacy", "Credit Hours": 3, "Prerequisites": "Pharmaceutical Analytical Chemistry I"},
            7: {"Course Code": "PG 202", "Course Name": "Pharmacognosy I", "Credit Hours": 3, "Prerequisites": "Medicinal Plants"},
            8: {"Course Code": "MD 204", "Course Name": "Psychology", "Credit Hours": 1, "Prerequisites": "Registration"}
        }
    },
    "Level 2": {
        "Semester 3": {
            1: {"Course Code": "PC 305", "Course Name": "Pharmaceutical Analytical Chemistry III", "Credit Hours": 2, "Prerequisites": "Pharmaceutical Analytical Chemistry II"},
            2: {"Course Code": "PC 306", "Course Name": "Pharmaceutical Organic Chemistry III", "Credit Hours": 3, "Prerequisites": "Pharmaceutical Organic Chemistry II"},
            3: {"Course Code": "PG 303", "Course Name": "Pharmacognosy II", "Credit Hours": 3, "Prerequisites": "Pharmacognosy I"},
            4: {"Course Code": "MD 305", "Course Name": "Physiology and Pathophysiology", "Credit Hours": 3, "Prerequisites": "Anatomy and Histology"},
            5: {"Course Code": "PT 303", "Course Name": "Pharmaceutics I", "Credit Hours": 3, "Prerequisites": "Physical Pharmacy"},
            6: {"Course Code": "NP 303", "Course Name": "Scientific Writing", "Credit Hours": 2, "Prerequisites": "Registration"},
            7: {"Course Code": "PHI3001", "Course Name": "Scientific Thinking", "Credit Hours": 1, "Prerequisites": "Information Technology"},
            8: {"Course Code": "HST3001", "Course Name": "Sinai History", "Credit Hours": 1, "Prerequisites": "Registration"}
        },
        "Semester 4": {
            1: {"Course Code": "PB 401", "Course Name": "Biochemistry I", "Credit Hours": 3, "Prerequisites": "Pharmaceutical Organic Chemistry III"},
            2: {"Course Code": "PM 401", "Course Name": "General Microbiology and Immunology", "Credit Hours": 3, "Prerequisites": "Registration"},
            3: {"Course Code": "PC 407", "Course Name": "Instrumental Analysis", "Credit Hours": 3, "Prerequisites": "Pharmaceutical Analytical Chemistry III"},
            4: {"Course Code": "MD 406", "Course Name": "Pathology", "Credit Hours": 2, "Prerequisites": "Physiology and Pathophysiology"},
            5: {"Course Code": "PT 404", "Course Name": "Pharmaceutics II", "Credit Hours": 3, "Prerequisites": "Pharmaceutics I"},
            6: {"Course Code": "NP 404", "Course Name": "Presentation & Communication Skills", "Credit Hours": 2, "Prerequisites": "Information Technology"},
            7: {"Course Code": "PO 401", "Course Name": "Biostatistics", "Credit Hours": 1, "Prerequisites": "Registration"}
        }
    },
    "Level 3": {
        "Semester 5": {
            1: {"Course Code": "PB 502", "Course Name": "Biochemistry II", "Credit Hours": 3, "Prerequisites": "Biochemistry I"},
            2: {"Course Code": "PM 502", "Course Name": "Pharmaceutical Microbiology", "Credit Hours": 3, "Prerequisites": "General Microbiology and Immunology"},
            3: {"Course Code": "PG 504", "Course Name": "Phytochemistry", "Credit Hours": 3, "Prerequisites": "Pharmacognosy II"},
            4: {"Course Code": "PT 505", "Course Name": "Pharmaceutics III", "Credit Hours": 3, "Prerequisites": "Pharmaceutics II"},
            5: {"Course Code": "PC 508", "Course Name": "Medicinal Chemistry I", "Credit Hours": 3, "Prerequisites": "Pharmaceutical Organic Chemistry III"},
            6: {"Course Code": "PO 502", "Course Name": "Pharmacology I", "Credit Hours": 3, "Prerequisites": "Physiology and Pathophysiology"}
        },
        "Semester 6": {
            1: {"Course Code": "PM 603", "Course Name": "Parasitology", "Credit Hours": 3, "Prerequisites": "Registration"},
            2: {"Course Code": "PT 606", "Course Name": "Biopharmaceutics & Pharmacokinetics", "Credit Hours": 3, "Prerequisites": "Pharmaceutics III, Mathematics"},
            3: {"Course Code": "PG 605", "Course Name": "Applied and Forensic Pharmacognosy", "Credit Hours": 2, "Prerequisites": "Phytochemistry"},
            4: {"Course Code": "PT 607", "Course Name": "Pharmaceutics IV", "Credit Hours": 3, "Prerequisites": "Pharmaceutics III"},
            5: {"Course Code": "PO 603", "Course Name": "Pharmacology II", "Credit Hours": 3, "Prerequisites": "Pharmacology I"},
            6: {"Course Code": "PC 609", "Course Name": "Medicinal Chemistry II", "Credit Hours": 3, "Prerequisites": "Medicinal Chemistry I"}
        }
    },
    "Level 4": {
        "Semester 7": {
            1: {"Course Code": "PM 704", "Course Name": "Medical Microbiology", "Credit Hours": 3, "Prerequisites": "Pharmaceutical Microbiology"},
            2: {"Course Code": "PO 704", "Course Name": "Pharmacology III", "Credit Hours": 3, "Prerequisites": "Pharmacology II"},
            3: {"Course Code": "PC 710", "Course Name": "Drug Design", "Credit Hours": 2, "Prerequisites": "Medicinal Chemistry II"},
            4: {"Course Code": "PB 703", "Course Name": "Clinical Biochemistry", "Credit Hours": 3, "Prerequisites": "Biochemistry II"},
            5: {"Course Code": "PT 708", "Course Name": "Pharmaceutical Technology I", "Credit Hours": 3, "Prerequisites": "Pharmaceutics IV"},
            6: {"Course Code": "NP 705", "Course Name": "Pharmaceutical Legislations & Professional Ethics", "Credit Hours": 1, "Prerequisites": "Registration"},
            7: {"Course Code": "PE ---", "Course Name": "Elective", "Credit Hours": 2, "Prerequisites": "Registration"}
        },
        "Semester 8": {
            1: {"Course Code": "PP 801", "Course Name": "Clinical Pharmacokinetics", "Credit Hours": 3, "Prerequisites": "Biopharmaceutics and Pharmacokinetics"},
            2: {"Course Code": "PO 805", "Course Name": "Drug Information", "Credit Hours": 2, "Prerequisites": "Pharmacology III"},
            3: {"Course Code": "PO 806", "Course Name": "Toxicology & Forensic Chemistry", "Credit Hours": 3, "Prerequisites": "Pharmacology III"},
            4: {"Course Code": "PP 802", "Course Name": "Hospital Pharmacy", "Credit Hours": 2, "Prerequisites": "Pharmacology III"},
            5: {"Course Code": "PT 809", "Course Name": "Pharmaceutical Technology II", "Credit Hours": 3, "Prerequisites": "Pharmaceutical Technology I"},
            6: {"Course Code": "PP 803", "Course Name": "Clinical Pharmacy and Pharmacotherapeutics I", "Credit Hours": 3, "Prerequisites": "Pharmacology III"},
            7: {"Course Code": "PE ---", "Course Name": "Elective", "Credit Hours": 2, "Prerequisites": "Registration"}
        }
    },
    "Level 5": {
        "Semester 9": {
            1: {"Course Code": "PB 901", "Course Name": "Biotechnology", "Credit Hours": 3, "Prerequisites": "Biochemistry II"},
            2: {"Course Code": "PP 902", "Course Name": "Community Pharmacy Practice", "Credit Hours": 2, "Prerequisites": "Clinical Pharmacy and Pharmacotherapeutics I"},
            3: {"Course Code": "PP 903", "Course Name": "Public Health", "Credit Hours": 2, "Prerequisites": "Registration"},
            4: {"Course Code": "PG 904", "Course Name": "Phytotherapy and Aromatherapy", "Credit Hours": 2, "Prerequisites": "Phytochemistry"},
            5: {"Course Code": "PT 905", "Course Name": "Good Manufacturing Practice", "Credit Hours": 2, "Prerequisites": "Pharmaceutical Technology II"},
            6: {"Course Code": "NP 906", "Course Name": "Marketing & Pharmacoeconomics", "Credit Hours": 2, "Prerequisites": "Registration"},
            7: {"Course Code": "PP 907", "Course Name": "Clinical Pharmacy and Pharmacotherapeutics II", "Credit Hours": 3, "Prerequisites": "Clinical Pharmacy and Pharmacotherapeutics I"},
            8: {"Course Code": "PE ---", "Course Name": "Elective", "Credit Hours": 2, "Prerequisites": "Registration"}
        },
        "Semester 10": {
            1: {"Course Code": "PC 1001", "Course Name": "Quality Control of Pharmaceuticals", "Credit Hours": 3, "Prerequisites": "Instrumental Analysis"},
            2: {"Course Code": "PP 1002", "Course Name": "First Aid", "Credit Hours": 2, "Prerequisites": "Registration"},
            3: {"Course Code": "PO 1003", "Course Name": "Drug Interaction", "Credit Hours": 2, "Prerequisites": "Pharmacology III"},
            4: {"Course Code": "PT 1004", "Course Name": "Advanced Drug Delivery Systems", "Credit Hours": 2, "Prerequisites": "Pharmaceutical Technology II"},
            5: {"Course Code": "PP 1005", "Course Name": "Clinical Pharmacy and Pharmacotherapeutics III", "Credit Hours": 3, "Prerequisites": "Clinical Pharmacy and Pharmacotherapeutics II"},
            6: {"Course Code": "NP 1006", "Course Name": "Entrepreneurship", "Credit Hours": 2, "Prerequisites": "Registration"},
            7: {"Course Code": "PP 1007", "Course Name": "Clinical Research, Pharmacoepidemiology & Pharmacovigilance", "Credit Hours": 2, "Prerequisites": "Clinical Pharmacy and Pharmacotherapeutics II"},
            8: {"Course Code": "PE ---", "Course Name": "Elective", "Credit Hours": 2, "Prerequisites": "Registration"}
        }
    }
}

pharmacy_regulations = {
    "Program Overview": {
        "Article 1": "The PharmD program aims to develop pharmaceutical sciences and healthcare solutions.",
        "Article 2": "Graduates receive a Bachelor of Pharmacy (PharmD) degree under a credit-hour system.",
        "Article 3": "The program qualifies graduates for professional licensing and postgraduate studies."
    },
    "Study System & Course Registration": {
        "Article 4": "The program spans five years of coursework plus one internship year.",
        "Article 5": "A student must complete 100 hours of internship training before graduation.",
        "Article 6": "Each academic year includes two semesters (15 weeks each) with an optional summer term.",
        "Article 7": "Maximum 21 credit hours per semester; course withdrawal is allowed within deadlines.",
        "Article 22": "Student couldn't register any course without finishing all its prerequisites",
        "Article 23": "If the student doesn't meet the couses requirements, then should first register the requirements on next semester"
    },
    "Attendance & Language of Instruction": {
        "Article 8": "Students must attend at least 65% of classes to qualify for final exams.",
        "Article 16": "Students must attend at least 65% of classes to qualify for final exams.",
        "Article 18": "The primary language of instruction is English, with some courses offered in Arabic."
    },
    "Internship & Graduation Requirements": {
        "Article 9": "The internship year includes hands-on training in hospitals, pharmacies, and companies.",
        "Article 14": "Internship follows a structured rotational training program in various pharmacy fields.",
        "Article 15": "Graduation requires completion of all coursework, passing the internship, and a graduation project."
    },
    "Exams & Evaluation": {
        "Article 17": "Final exams follow the university calendar. Students missing exams without a valid excuse fail the course.",
        "Article 19": "Grades are based on midterms, finals, participation, and practical assessments."
    },
    "Student Conduct & Disciplinary Actions": {
        "Article 10": "Students must follow the university’s disciplinary regulations.",
        "Article 20": "Violations (cheating, plagiarism, disruptive behavior) can lead to suspension or expulsion."
    },
    "Academic Program Updates": {
        "Article 11": "The study plan and course requirements are defined in the official curriculum.",
        "Article 12": "Course content is specified in the curriculum and can be updated as needed.",
        "Article 13": "Courses can be updated by up to 20% upon faculty approval.",
        "Article 21": "The curriculum is regularly revised to align with scientific advancements and accreditation standards."
    }
}

# CGPA input
st.header('Please enter your CGPA')
cgpa = st.number_input('Enter you CGPA', min_value=0.0, max_value=4.0)
print(cgpa)

# Additional Notes
st.header('If you have any additional notes')
st.text_area('Enter your notes here')

prompt = f'''
You are an academic advisor helping a university student with course registration. 
Your task is to analyze the student's academic history and provide a personalized list of courses they should register for the upcoming semester.

Inputs You Have:
1. University Course Catalog dictionary {pharmacy_courses}:
   A list of all available courses in all study years. Each course includes:
       Course Code
       Course Name
       Credit Hours
       Prerequisites
        
2. University Regulations summairzed into dictionary {pharmacy_regulations}

Now, you have the following details for a student. The student registed {student_selected_courses} courses only. The student's CGPA is {cgpa}. The student asked you about what courses should be registed so that doesn't conflict with the university regulations. Based on the student CGPA and completed courses do the following:
1. If the CGPA is below 2 then the student could register 12 cridet hours only
2. If the CGPA is above 2 then do the following:
    2.1 Check the completed courses
    2.2 Check the courses should be registered:
        2.2.2 if the student completed all its prerequisites then the student can register the course
        2.2.2 if the student doesn't meet all course prerequisites then the student cannot register the course and should finish the prerequisites first. So, the student will register the prerequisites in the next semester and the the next courses with completed prerequisites only.
3.If the student haven't registered any courses yet, then should register courses in semester one regarless the student's CGPA
'''

def guide_me():
    json_response = get_response(prompt)
    list_of_courses = json_response.list_of_courses
    advice = json_response.advice
    
    results_container.markdown(f'*{advice}*')

    for i in range(len(list_of_courses)):
        course = list_of_courses[i]
        with results_container.expander(f'📚️ {course.course_name} {course.credit_hours} Credit Hours'):
            st.markdown(f'*{course.course_code}*')
            st.write(f'Prerequisites: {course.prerequisites}')
            st.write(f'Course Description: {course.course_description}')
        
    
# generate button
st.button('Guide Me 🚀️', type = 'primary', on_click = guide_me , use_container_width = True)

# Results
st.markdown('---')
st.header('Here is your Guide 🤓️')

results_container = st.container()



# Sidebar
def response_generator(prompt):
    response = chat_message(prompt)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
        

with st.sidebar:
    st.title('UniGuide Chat 🤖️')
    
    # containers
    messages_container = st.container()
    chat_input_container = st.container()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with messages_container.chat_message(message["role"]):
            messages_container.markdown(message["content"])
    
    # Accept user input
    if prompt_sbar := chat_input_container.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt_sbar})
        # Display user message in chat message container
        with messages_container.chat_message("user"):
            messages_container.markdown(prompt_sbar)
    
        # Display assistant response in chat message container
        with messages_container.chat_message("assistant"):
            response = messages_container.write_stream(response_generator(prompt_sbar))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    
        
    
        





    



