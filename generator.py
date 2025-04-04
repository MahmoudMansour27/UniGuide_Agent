from google import genai
from pydantic import BaseModel


# parameters
model_flash = 'gemini-2.0-flash'
api_key_flash = 'sk'

client = genai.Client(api_key = api_key_flash)

class Course(BaseModel):
    course_code: str
    course_name: str
    credit_hours: int
    prerequisites: str
    course_description: str
    

class Registration_courses(BaseModel):
    list_of_courses: list[Course]
    advice: str
    

def get_response(prompt):
    response = client.models.generate_content(
            model = model_flash,
            contents= prompt,
            config={
                    'response_mime_type': 'application/json',
                    'response_schema': Registration_courses
                }
    )
    return response.parsed


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
        "Article 5": "A student must complete 300 hours of internship training before graduation.",
        "Article 6": "Each academic year includes two semesters (15 weeks each) with an optional summer term.",
        "Article 7": "Maximum 21 credit hours per semester; course withdrawal is allowed within deadlines."
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

cgpa = 3.8

student_selected_courses = ["Pharmaceutical Analytical Chemistry I",
"Pharmaceutical Organic Chemistry I",
"Pharmacy Orientation",
"English",
"Medicinal Plants",
"Information Technology",
"Mathematics",
"Human Rights and Fighting Corruption"]

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

Now, you have the following details for a student. The student registed {student_selected_courses} courses. 
The student's CGPA is {cgpa}. The student asked you about what courses should be registed so that doesn't conflict with university Regulations 
and doesn't affect the student's success.
'''

#Your Response structure should be {response_structure} as a JSON file.

#res = get_response(prompt)

# chat
chat = client.chats.create(model='gemini-2.0-flash')
def chat_message(msg):
    response = chat.send_message(
            message=msg
        )
    return response.text

#print(chat_message('tell me 5 Egyptain male names').text)

