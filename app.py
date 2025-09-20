import os, io, re, json, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import pdfplumber
except:
    pdfplumber = None
try:
    import docx
except:
    docx = None

try:
    from langchain import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain import LLMChain
except:
    OpenAI = None

# ---------------------- Page Config (App Icon & Title) ----------------------
ICON_PATH = os.path.join(os.path.dirname(__file__), "icon.png")
if os.path.exists(ICON_PATH):
    page_icon = ICON_PATH
else:
    page_icon = "📝"  # fallback emoji icon

st.set_page_config(
    page_title="Automated Resume Relevance Check",
    page_icon="icon.png",
    layout="wide"
)

# ---------------------- Helpers ----------------------
def extract_text(uploaded_file):
    data = uploaded_file.read()
    fname = uploaded_file.name.lower()
    if fname.endswith('.pdf') and pdfplumber:
        text = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    elif (fname.endswith('.docx') or fname.endswith('.doc')) and docx:
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    else:
        try:
            return data.decode('utf-8')
        except:
            return data.decode('latin-1', errors='ignore')

def clean_text(t: str) -> str:
    return re.sub(r'\s+', ' ', t).strip()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_basic_info(text: str) -> dict:
    email = re.search(r'[\w\.-]+@[\w\.-]+', text)
    phone = re.search(r'\+?\d[\d\s\-]{7,}\d', text)
    return {
        'email': email.group() if email else 'N/A',
        'phone': phone.group() if phone else 'N/A'
    }

# ---------------------- Resume Parsing ----------------------
def extract_education(resume_text):
    text = normalize_text(resume_text)
    degree_patterns = [
        r'bachelor of [a-z ]+', r'b\.sc', r'bsc', r'b\.tech', r'btech', r'be', r'b\.e',
        r'ba', r'bcom', r'master of [a-z ]+', r'm\.sc', r'msc', r'm\.tech', r'mtech', r'mba',
        r'ma', r'phd', r'doctorate'
    ]
    matches = []
    for pattern in degree_patterns:
        found = re.findall(pattern, text, re.IGNORECASE)
        matches.extend(found)
    return list(set(matches))

def extract_experience(resume_text):
    keywords = ['intern', 'developer', 'engineer', 'manager', 'analyst', 'consultant']
    lines = resume_text.lower().split('\n')
    return [line.strip() for line in lines if any(k in line for k in keywords)]

def extract_certifications(resume_text):
    keywords = ['certified', 'certificate', 'aws', 'gcp', 'azure', 'pmp', 'scrum']
    lines = resume_text.lower().split('\n')
    return [line.strip() for line in lines if any(k in line for k in keywords)]

# ---------------------- LLM Scoring ----------------------
SCORE_PROMPT_TEMPLATE = '''Job Description:
{jd}

Candidate Resume:
{resume}

Respond in JSON including:
{
  "relevance_score":0-100,
  "strengths":[],
  "missing_skills":[],
  "summary":"",
  "education":[],
  "experience":[],
  "certifications":[],
  "skills":[]
}'''

def call_llm_for_scoring(jd: str, resume: str) -> dict:
    jd = clean_text(jd)
    resume = clean_text(resume)
    skills_weights = {
        'Python': 15, 'Flask': 10, 'Django': 10, 'REST': 10,
        'SQL': 10, 'NoSQL': 10, 'AWS': 10, 'Git': 5,
        'HTML': 5, 'CSS': 5, 'JavaScript': 10, 'React': 10
    }

    if OpenAI and os.getenv('OPENAI_API_KEY'):
        prompt = SCORE_PROMPT_TEMPLATE.format(jd=jd[:4000], resume=resume[:8000])
        llm = OpenAI(temperature=0)
        chain = LLMChain(llm=llm, prompt=PromptTemplate(template=SCORE_PROMPT_TEMPLATE, input_variables=['jd','resume']))
        try:
            raw = chain.run({'jd': jd, 'resume': resume})
            data = json.loads(re.search(r'{.*}', raw, re.DOTALL).group())
            return data
        except:
            pass

    matched = []
    total_score = 0
    for skill, weight in skills_weights.items():
        if skill.lower() in resume.lower():
            matched.append(skill)
            total_score += weight
    total_score = min(100, total_score)
    missing = [s for s in skills_weights if s not in matched]
    basic_info = extract_basic_info(resume)

    if total_score >= 80:
        summary_text = f"Strong candidate with key skills: {', '.join(matched)}."
    elif total_score >= 50:
        summary_text = f"Moderate fit. Strengths: {', '.join(matched)}; Missing: {', '.join(missing)}."
    else:
        summary_text = f"Low fit. Only a few relevant skills: {', '.join(matched)}."

    result_experience = extract_experience(resume)
    result_education = extract_education(resume)
    result_certifications = extract_certifications(resume)

    return {
        'relevance_score': total_score,
        'strengths': matched,
        'missing_skills': missing,
        'summary': summary_text,
        'skills': matched,
        'experience': result_experience,
        'education': result_education,
        'certifications': result_certifications,
        'email': basic_info['email'],
        'phone': basic_info['phone'],
        'raw': ''
    }

# ---------------------- Streamlit UI ----------------------
# ---------------------- Streamlit UI ----------------------
from PIL import Image

logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=120)  

st.title("Automated Resume Relevance Check System - Advanced Version")

st.subheader("Job Description")
jd_text = st.text_area("Paste Job Description", height=250)
jd_file = st.file_uploader("Or upload JD file", type=['pdf','docx','txt'])
if jd_file is not None:
    jd_text = extract_text(jd_file)

st.subheader("Candidate Resumes")
uploaded_files = st.file_uploader("Upload one or more resumes", type=['pdf','docx','txt'], accept_multiple_files=True)

company_requirements = {
    "Startup": ["Python", "Flask", "Django", "REST", "Git"],
    "MNC": ["Python", "Flask", "Django", "REST", "SQL", "NoSQL", "AWS", "Git"],
    "Product Company": ["Python", "Django", "REST", "AWS", "Git"]
}
eligibility_threshold = 0.8

results_list = []

if st.button("Check Relevance"):
    if not jd_text.strip():
        st.error("Please provide a Job Description.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        for f in uploaded_files:
            resume_text = extract_text(f)
            result = call_llm_for_scoring(jd_text, resume_text)
            result['candidate_name'] = f.name

            eligible_companies = []
            not_eligible_companies = []
            for company, skills_required in company_requirements.items():
                matched_skills = set(result.get('skills', []))
                required_skills = set(skills_required)
                match_ratio = len(matched_skills & required_skills) / len(required_skills)
                if match_ratio >= eligibility_threshold:
                    eligible_companies.append(company)
                else:
                    not_eligible_companies.append(company)
            result['eligible'] = "Yes" if eligible_companies else "No"
            result['eligible_companies'] = ", ".join(eligible_companies) if eligible_companies else "None"
            result['not_eligible_companies'] = ", ".join(not_eligible_companies) if not_eligible_companies else "None"

            results_list.append(result)

        if results_list:
            df = pd.DataFrame(results_list)
            st.subheader("Candidate Comparison Table")
            display_df = df[['candidate_name','relevance_score','eligible','eligible_companies','skills']]
            display_df = display_df.sort_values(by='relevance_score', ascending=False)
            st.dataframe(display_df)

            top_score = display_df['relevance_score'].max()
            st.markdown(f"**Top Candidate(s) Score:** {top_score}")

            # ---------------------- Charts for Skills ----------------------
            st.subheader("Skills Coverage Charts")
            for res in results_list:
                st.markdown(f"**{res['candidate_name']}**")

                # Bar Graph (Skills vs Missing)
                skills = res.get('skills', [])
                missing = res.get('missing_skills', [])
                plt.figure(figsize=(6,3))
                plt.bar(skills, [1]*len(skills), color='green', label='Acquired')
                plt.bar(missing, [1]*len(missing), color='red', label='Missing')
                plt.ylabel("Skill Presence")
                plt.legend()
                st.pyplot(plt)
                plt.close()

                # Pie Chart
                acquired = len(skills)
                missing_count = len(missing)
                labels = ['Acquired', 'Missing']
                sizes = [acquired, missing_count]
                colors = ['green', 'red']
                fig1, ax1 = plt.subplots()
                ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                ax1.axis('equal')
                st.pyplot(fig1)
                plt.close(fig1)

            # ---------------------- Individual Candidate Details ----------------------
            st.subheader("Detailed Candidate Analysis")
            for res in results_list:
                with st.expander(res['candidate_name']):
                    badge_color = "green" if res['eligible']=="Yes" else "red"
                    st.markdown(f"<span style='color:{badge_color}; font-weight:bold'>Eligibility: {res['eligible']}</span>", unsafe_allow_html=True)
                    st.write("Email:", res.get('email','N/A'))
                    st.write("Phone:", res.get('phone','N/A'))
                    st.write("Skills:", res.get('skills',[]))
                    st.write("Missing Skills:", res.get('missing_skills',[]))
                    st.write("Experience:", res.get('experience',[]))
                    st.write("Education:", res.get('education',[]))
                    st.write("Certifications:", res.get('certifications',[]))
                    st.metric("Relevance Score", f"{res.get('relevance_score','N/A')}/100")
                    st.write("Summary:", res.get('summary',''))
                    st.write("Eligible Companies:", res['eligible_companies'])
                    st.write("Not Eligible Companies:", res['not_eligible_companies'])

            # ---------------------- CSV Download ----------------------
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Results as CSV", data=csv, file_name='resume_results.csv', mime='text/csv')
