from crewai import Agent, Task, Crew, LLM, Process
from pydantic import BaseModel
import os
import json
from knowledge import pharmacy_regulations, pharmacy_semesters_credit_hours
from knowledge import semester_courses_codes, key_courses_codes, credits_codes
from prerequisite_checker import eligiablitiy_filter
import time


# llms
os.environ['GROQ_API_KEY'] = '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'

llama_llm = LLM(
    model = "groq/llama-3.3-70b-versatile",
    temperature= 0.0
)

gemma_llm = LLM(
    model = "groq/gemma2-9b-it",
    temperature= 0.0
)

llama_instant_llm = LLM(
    model = "groq/llama-3.1-8b-instant",
    temperature= 0.0
)



# Rules Agent ------------------------------
class credit_JSON(BaseModel):
  english_course: str
  current_semester: int
  registration_semester: int
  ordinary_registration_semester_credit_hours: int
  student_maximum_credit_hours: int
  reasoning: str


rules_agent = Agent(
    role="University Academic Advisor specializing in course registration policies.",
    goal=(
        "Evaluate a student's CGPA ({cgpa}), and English proficiency level ({eng_level}) "
        "to determine that the student is currently at semester ({curr_semester}), maximum allowed credit hours, and English course requirement. "
        "Ensure all recommendations comply with university regulations ({regulations}) and support academic success. "
        "Store the results in a structured JSON format for record-keeping and further processing."
    ),
    backstory=(
        "As an experienced university academic advisor, you specialize in course registration policies, credit hour limitations, "
        "and language proficiency requirements. You assess students' academic progress based on their CGPA, English level, "
        "and completed courses to provide precise registration recommendations. Your structured approach ensures compliance with "
        "university regulations while optimizing student success. You document your recommendations in JSON format for easy "
        "integration with university systems."
    ),
    llm=llama_llm,
)

rules_task = Task(
    description=(
        "Assess the student's academic standing based on CGPA ({cgpa}) and English proficiency level ({eng_level}). "
        "Determine that the student is currently at semester ({curr_semester}) and set the next one as their registration semester. "
        "Based on the registration semester credit limits ({sem_credits}) and CGPA, calculate the maximum number of credit hours allowed. "
        "Check if an English course is required according to university regulations ({regulations}). "
    ),
    expected_output=(
        "A structured JSON file containing:\n"
        "- Whether the student needs to take an English course.\n"
        "- Current semester.\n"
        "- Registration semester.\n"
        "- Ordinary registration semester credit hours based semesters credit hour limits ({sem_credits}).\n"
        "- The maximum credit hours the student can register based on both CGPA and if there any registration semester exception such that in the semester 2 (per Article 27).\n"
        "- A justification for the decision aligned with university policies.\n"
    ),
    agent=rules_agent,
    output_json=credit_JSON,
    output_file="/outputs/credits.json",
)

# priority scores -----------------------------------------------------
class Prioritisied(BaseModel):
  prioritisied_courses: list
  reasoning: str

courses_prioritizer = Agent(
    role = 'University Course Advisor specializing in course prioritization and academic planning',
    goal = ' '.join([
        "Assign the student's registration shortlist ({shortlist_cou}) with a priority score."
        "You can refer to the university's key courses as a guide for prioritization {key_courses}.",
        "Key courses are the most critical courses in the university curriculum.",
        "Generate a structured JSON file containing the prioritized course list to guide students in making informed course registration decisions."
    ]),
    backstory = ' '.join([
        "As an expert in academic advising and course sequencing, you focus on evaluating course priorities to optimize students\' academic progress.",
        "You assess the importance of each course within the university curriculum and its impact on future course eligibility, using university-defined priorities [Next Semester and key courses: 5 Very Very High Priority, Next Level and key courses: 4 Very High Priority, Next Semester courses: 3 High Priority, Key courses: 2 Medium Priority, Other courses: 1 Low Priority].",
        "Your methodical approach ensures that students register for the most impactful courses first, preventing scheduling conflicts and delays in degree progression.",
        "All your recommendations are systematically stored in JSON format, which streamlines academic planning and decision-making for both advisors and students."
    ]),
    llm = llama_llm,
)

rank_courses_task = Task(
    description= ' '.join([
        "Rank student's registration shortlist ({shortlist_cou}) using one of the following two methods only:\n",
        "Method number 1:\n",
        "If the student's credit hour limit {credits} satisfies the total credit hours of the registration semester {reg_sem_credit},",
        "then assign courses from the student's eligible course list {shortlist_cou} that in {reg_sem_courses} according to [Key Course: 7, Not a Key Course: 6]. You can use ({key_courses}) as a reference of key courses.\n", 
        "For the other courses that not in the list assign he priority criteria based on the following criteria [Next Semester and key courses: 5 Very Very High Priority, Next Level and key courses: 4 Very High Priority, Next Semester courses: 3 High Priority, Key courses: 2 Medium Priority, Other courses: 1 Low Priority].",
        
        "Method number 2:\n",
        "If the student's credit hour limit {credits} does not meet the total credit hours for registration semester {reg_sem_credit},",
        "rank all the student's eligible courses {shortlist_cou} based on university-defined priorities [Next Semester and key courses: 5 Very Very High Priority, Next Level and key courses: 4 Very High Priority, Next Semester courses: 3 High Priority, Key courses: 2 Medium Priority, Other courses: 1 Low Priority] only and don't assign with priority 6 even it's in the list.",
        
        "Use the current semester ({curr_sem}) to reference the next semester ({next_sem}) and next level courses when assigning priority correctly.",
        "This ensures that students register for the most crucial courses first, supporting their academic progression."
    ]),
    expected_output = '\n'.join([
        "A JSON file containing:",
        "- A list of eligible courses with assigned priority scores.",
        "- A detailed explanation of the ranking methodology used."
    ]),
    agent=courses_prioritizer,
    output_json=Prioritisied,
    output_file='/outputs/ranked.json'
)

# selection agent ------------------------------------------------
class Selected(BaseModel):
  selected_courses: list
  reasoning: str
  total_credit_hours: int

courses_selector = Agent(
    role = 'University Course Selection Advisor specializing in optimized course scheduling',
    goal = ' '.join(["Select the most suitable courses from prioritized couses list {prioritized_list} for a student based on their credit hour limit {crdits_limit}, course priority, and English course requirement {english_course}.",
                     "You can access all university courses credit hours {credits} as a reference.",
                     "The student wants your help to register the next semester courses.",
                     "Ensure an optimal selection that maximizes academic progress while adhering to university policies.",
                     "Try to assign the maximum number of allowed credit hours of the student."
                     "Generate a structured JSON file containing the final course list with a justification for each selection."]),
    backstory=' '.join(["With extensive experience in academic advising and course scheduling, you specialize in balancing", 
                        "course priorities with credit hour constraints to create the most effective registration plan for students who want your help to register the next semester courses.", 
                        "Your expertise ensures that students enroll in the most impactful courses first while fulfilling necessary requirements.",
                        "By systematically analyzing available options, you generate a well-reasoned course selection, saved in JSON format,",
                        "to help students and advisors make informed decisions."]),
    llm = llama_instant_llm,
)

select_courses_task = Task(
    description= ' '.join(["Choose courses with the highest priorities based on their course priority rankings {prioritized_list}, credit",
                            "hour limit ({crdits_limit}), and English course requirement ({english_course}).",
                            "Ensure that the english course credit hours is included in the student's credit hour limit {crdits_limit}",
                            "English courses is mandatory and has the highest priority so it is should be registrated first.",
                            "Ensure the selection maximizes academic progress and assign as max as possible credit hours."]),
    expected_output = '\n'.join(["A JSON file containing:",
                                 "- The final list of selected courses for the student to register with course credit hours and reasoning.",
                                 "- A description justifying why each course was chosen."]),
    agent=courses_selector,
    output_json=Selected,
    output_file='/outputs/selected.json'
)


# crews creation -------------------------------------------
rules_crew = Crew(
  agents=[rules_agent],
  tasks=[rules_task],
  process= Process.sequential
)

priority_crew = Crew(
  agents=[courses_prioritizer],
  tasks=[rank_courses_task],
  process= Process.sequential
)

selection_crew = Crew(
  agents=[courses_selector],
  tasks=[select_courses_task],
  process= Process.sequential
)

files = [
  './outputs/credits.json',
  './outputs/ranked.json',
  './outputs/selected.json',
]

def run(cgpa, eng_lvl, curr_sem, comp_courses):
  # delete_files
  if os.path.exists(files[0]):
    os.remove(files[0])
    print('credits has been deleted')

  if os.path.exists(files[1]):
    os.remove(files[1])
    print('ranked has been deleted')


  if os.path.exists(files[2]):
    os.remove(files[2])
    print('selected has been deleted')


  rules_crew.kickoff(
    inputs={
      'cgpa': cgpa,
      'eng_level': eng_lvl,
      'regulations': pharmacy_regulations,
      'curr_semester': curr_sem,
      'sem_credits': pharmacy_semesters_credit_hours,
    }
  )
  print('rules crew kickoff done')

  with open(files[0]) as credit_json:
    credit_json_content = json.load(credit_json)


  time.sleep(3)
  priority_crew.kickoff(
    inputs={
      'credits':credit_json_content['student_maximum_credit_hours'],
      'reg_sem_credit':credit_json_content['ordinary_registration_semester_credit_hours'],
      'shortlist_cou': eligiablitiy_filter(comp_courses),
      'reg_sem_courses': semester_courses_codes[credit_json_content['registration_semester']], 
      'curr_sem': credit_json_content['current_semester'],
      'next_sem': credit_json_content['registration_semester'],
      'key_courses': key_courses_codes
    }
  )
  print('priority crew kickoff done')


  with open(files[1]) as prioritise_json:
    prioritise_json_content = json.load(prioritise_json)

  selection_crew.kickoff(
    inputs={
      'prioritized_list': prioritise_json_content['prioritisied_courses'],
      'crdits_limit': credit_json_content['student_maximum_credit_hours'],
      'english_course': credit_json_content['english_course'],
      'credits': credits_codes,
    }
  )
  print('selection crew kickoff done')
  print('Agent has finished its task!!!')


#test
# from knowledge import completed_courses
# student_cgpa = 3.2
# student_eng_lvl = 4
# student_curr_sem = 1
# run(cgpa=student_cgpa, eng_lvl=student_eng_lvl, curr_sem=student_curr_sem, comp_courses=completed_courses)

