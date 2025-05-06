class PromptRegistry:
    """Registry of prompt templates for LLM interactions.

    This class maintains a collection of prompt templates used for different
    stages of the trial matching pipeline. Each template consists of a system
    prompt and a user prompt, which can be formatted with variables at runtime.

    The registry includes prompts for:
    1. Patient and trial summarization (detailed and condensed)
    2. Exclusion and inclusion filtering
    3. Final match scoring
    4. Domain-specific variants for different datasets (e.g., SIGIR2016)
    """
    def __init__(self):
        self.prompts = {
            # Existing prompts
            "patient_summary": self._patient_summary(),
            "trial_summary": self._trial_summary(),
            "patient_condensed": self._patient_condensed(),
            "trial_condensed": self._trial_condensed(),
            "exclusion_filter": self._exclusion_filter(),
            "inclusion_filter": self._inclusion_filter(),
            "final_match_scoring": self._final_match_scoring(),

            # SIGIR2016-specific prompts
            "patient_summary_sigir2016": self._patient_summary_sigir2016(),
            "patient_condensed_sigir2016": self._patient_condensed_sigir2016(),
            "trial_summary_sigir2016": self._trial_summary_sigir2016(),
            "trial_condensed_sigir2016": self._trial_condensed_sigir2016(),
        }

    def get(self, name: str) -> dict:
        """Get prompt pair (system and user) by name.

        Args:
            name: Name of the prompt template to retrieve

        Returns:
            Dictionary containing 'system' and 'user' prompt templates
        """
        return self.prompts.get(name, {"system": "", "user": ""})

    def get_system(self, name: str) -> str:
        """Get just the system prompt for a template.

        Args:
            name: Name of the prompt template

        Returns:
            The system prompt string, or empty string if not found
        """
        prompt_pair = self.prompts.get(name, {})
        return prompt_pair.get("system", "")

    def get_user(self, name: str) -> str:
        """Get just the user prompt for a template.

        Args:
            name: Name of the prompt template

        Returns:
            The user prompt string, or empty string if not found
        """
        prompt_pair = self.prompts.get(name, {})
        return prompt_pair.get("user", "")

    @staticmethod
    def _patient_summary():
        return {
            "system": "You are a precision oncology specialist with expertise in extracting structured clinical information for trial matching. Your analysis is thorough, medically accurate, and organized for optimal patient-trial matching.",
            "user": """
    I need a structured summary of the following cancer patient record for clinical trial matching purposes.

    Patient Record:
    \"\"\"
    {patient_text}
    \"\"\"

    Extract and organize the key clinical information into these EXACT sections:

    DIAGNOSIS:
    • List each cancer diagnosis chronologically with dates
    • Include precise staging (TNM if available), grade, and anatomical location
    • Note histological subtype and any important pathology findings

    BIOMARKERS:
    • List all genomic alterations with specific mutations (e.g., BRAF V600E, EGFR exon 19 deletion)
    • Include biomarker expression levels with values when available (e.g., PD-L1 80%, HER2 3+)
    • Note testing methods and dates if mentioned

    TREATMENT HISTORY:
    • Chronological list of cancer therapies with start/end dates
    • Include specific regimens, doses if mentioned
    • Document response assessments (CR, PR, SD, PD) and duration
    • Note reason for discontinuation (progression, toxicity, completion)

    RELEVANT MEDICAL HISTORY:
    • Major comorbidities that could impact trial eligibility
    • Organ function issues (cardiac, hepatic, renal)
    • Prior surgeries or radiation treatments
    • Significant adverse events to previous treatments

    CURRENT STATUS:
    • Current ECOG/KPS performance status
    • Active symptoms and disease burden
    • Most recent disease assessment and progression status
    • Current laboratory values relevant to trial eligibility

    Focus ONLY on information explicitly stated in the record. Use bullet points for clarity. Do not infer or add details not present in the original text. If information for a section is not available, write "Not specified in record."
    """
        }

    @staticmethod
    def _patient_condensed():
        return {
            "system": "You are an expert in oncology trial matching who specializes in creating optimized patient representations for embedding-based retrieval systems. Your summaries focus exclusively on positive attributes relevant for trial matching while avoiding numerical thresholds and negations.",
            "user": """
    Create a CONDENSED PATIENT PROFILE (200-300 words) optimized for semantic matching with clinical trial criteria.

    Patient Record:
    \"\"\"
    {patient_text}
    \"\"\"

    TASK: Generate a single cohesive paragraph describing this cancer patient's key characteristics for trial matching. This text will be used for embedding-based similarity matching with trial descriptions.

    FOCUS EXCLUSIVELY ON:
    1. Cancer diagnoses with specific histology, molecular subtype, and staging
    2. Biomarkers and genetic mutations with expression status
    3. Key patient characteristics relevant to eligibility (functional status, organ function)
    4. Treatment history including lines of therapy and responses
    5. Current disease status and treatment needs

    CRITICAL GUIDELINES:
    • INCLUDE ONLY POSITIVE ATTRIBUTES that would qualify for trials
    • CONVERT ALL NUMERIC VALUES to descriptive clinical terms:
      ✓ "adult patient" instead of "52-year-old"
      ✓ "high PD-L1 expression" instead of "PD-L1 expression 80%"
      ✓ "adequate renal function" instead of "creatinine 1.1 mg/dL"
      ✓ "good performance status" instead of "ECOG 1"
    • USE PRECISE ONCOLOGY TERMINOLOGY consistently
    • WRITE AS A SINGLE COHERENT PARAGRAPH without subheadings
    • FOCUS ON CURRENT AND FACTUAL patient status
    • DO NOT include exclusion criteria or negative characteristics

    Your profile should enable optimal semantic similarity matching when compared with trial descriptions formatted in the same style.
    """
        }

    @staticmethod
    def _trial_summary():
        return {
            "system": "You are an oncology clinical trial expert specializing in protocol analysis and eligibility assessment. Your expertise lies in translating complex trial protocols into clear, structured criteria that can be matched against patient profiles.",
            "user": """
    Analyze the following clinical trial description and create a structured summary of eligibility criteria.

    Trial Protocol:
    \"\"\"
    {trial_text}
    \"\"\"

    Extract and organize the key information into these EXACT sections:

    DIAGNOSIS:
    • Target cancer type(s) and required histologies
    • Required disease stage or status (metastatic, locally advanced, etc.)
    • Specific molecular subtypes or histological features
    • Required measurable disease criteria (RECIST, etc.)

    BIOMARKERS:
    • Required biomarker status or expression levels
    • Genetic alterations or mutations required for eligibility
    • Testing requirements and thresholds
    • Molecular inclusion criteria

    TREATMENT HISTORY:
    • Required prior therapies or line of therapy
    • Treatment-free interval requirements
    • Prior response requirements
    • Restrictions on previous treatments

    EXCLUSION CRITERIA:
    • Medical conditions that disqualify participants
    • Prohibited concurrent medications
    • Contraindicated comorbidities or organ dysfunction
    • Other disqualifying factors

    INCLUSION CRITERIA:
    • Age and performance status requirements
    • Laboratory value thresholds (hematologic, chemistry)
    • Required organ function parameters
    • Other key eligibility factors

    Use bullet points for each criterion and focus on objective requirements. If information for a section is not specified in the protocol, write "Not specified in protocol." Do not add information not present in the original text.
    """
        }

    @staticmethod
    def _trial_condensed():
        return {
            "system": "You are an expert in clinical trial matching who specializes in creating optimized trial representations for embedding-based retrieval systems. Your summaries focus exclusively on positive eligibility attributes while avoiding numerical thresholds and negations.",
            "user": """
    Create a CONDENSED TRIAL PROFILE (200-300 words) optimized for semantic matching with cancer patients.

    Trial Protocol:
    \"\"\"
    {trial_text}
    \"\"\"

    TASK: Generate a single cohesive paragraph describing this trial's ideal candidate profile. This text will be used for embedding-based similarity matching with patient records.

    FOCUS EXCLUSIVELY ON:
    1. Target cancer type with specific histology, molecular subtype, and staging requirements
    2. Biomarker or genetic mutation requirements with expression status
    3. Patient characteristics sought (functional status, organ function requirements)
    4. Treatment history preferences or requirements
    5. Key eligibility factors unique to this trial

    CRITICAL GUIDELINES:
    • INCLUDE ONLY POSITIVE ATTRIBUTES that define eligible patients
    • CONVERT ALL NUMERIC THRESHOLDS to descriptive clinical terms:
      ✓ "adult patients" instead of "age > 18"
      ✓ "adequate neutrophil count" instead of "ANC > 1.5 × 10⁹/L"
      ✓ "normal kidney function" instead of "creatinine < 1.5 mg/dL"
      ✓ "good performance status" instead of "ECOG 0-1"
    • DO NOT MENTION ANY EXCLUSION CRITERIA or negative characteristics
    • USE PRECISE ONCOLOGY TERMINOLOGY consistently
    • WRITE AS A SINGLE COHERENT PARAGRAPH without subheadings

    Your profile should enable optimal semantic similarity matching when compared with patient records formatted in the same style.
    """
        }

    @staticmethod
    def _exclusion_filter():
        return {
            "system": "You are a precision oncology trial coordinator with extensive experience screening patients against exclusion criteria. You are known for your methodical analysis and ability to determine clear eligibility verdicts based on available clinical information.",
            "user": """
    Evaluate whether this patient should be EXCLUDED from the clinical trial based on the trial's exclusion criteria.

    Trial Exclusion Criteria:
    \"\"\"
    {exclusion_criteria}
    \"\"\"

    Patient Summary:
    \"\"\"
    {patient_summary}
    \"\"\"

    ANALYSIS INSTRUCTIONS:
    1. Examine each exclusion criterion carefully
    2. Compare each criterion against the patient's clinical information
    3. Only mark EXCLUDE if there is clear evidence that an exclusion criterion applies
    4. Give the patient the benefit of the doubt when information is ambiguous or missing

    YOUR EVALUATION PROCESS:
    • First, identify the key exclusion criteria categories (medical conditions, lab values, prior therapies, etc.)
    • For each category, check if the patient information contains relevant data
    • Determine if any criterion is definitely met based on explicit information
    • Consider both explicit statements and clearly implied clinical information

    IMPORTANT: Your response MUST follow this EXACT FORMAT to be properly processed:

    VERDICT: [EXCLUDE or PASS]

    REASON:
    [If EXCLUDE: Clearly state which specific exclusion criterion the patient meets, referencing the exact criterion language and the specific patient information that confirms it]
    [If PASS: State "Patient does not meet any exclusion criteria based on available information" or "Insufficient evidence to confirm any exclusion criteria"]

    Do not include any preliminary analysis, additional headers, or extra text. The formatted verdict and reason are required for automated processing.
    """
        }

    @staticmethod
    def _inclusion_filter():
        return {
            "system": "You are a clinical trial eligibility expert with specialized knowledge in oncology protocol assessment. You excel at determining whether patients meet specific inclusion criteria based on clinical evidence, and you understand the importance of structured, precise evaluations.",
            "user": """
    Evaluate whether this patient meets all inclusion criteria for the clinical trial.

    Trial Inclusion Criteria:
    \"\"\"
    {inclusion_criteria}
    \"\"\"

    Patient Summary:
    \"\"\"
    {patient_summary}
    \"\"\"

    ANALYSIS INSTRUCTIONS:
    1. Methodically assess each inclusion criterion
    2. Compare against the patient's documented information
    3. Determine if ALL criteria are met, SOME are uncertain, or ANY are definitely not met
    4. For uncertain criteria, note exactly what information is missing

    YOUR EVALUATION PROCESS:
    • First, categorize the inclusion criteria (diagnosis, biomarkers, prior therapy, functional status, etc.)
    • For each criterion, check if the patient data contains relevant information
    • Mark each criterion as: Confirmed Met, Likely Met, Cannot Determine, or Definitely Not Met
    • Consider whether missing information would be critical for determination
    • Make a final determination based on the overall assessment

    IMPORTANT: Your response MUST follow this EXACT FORMAT to be properly processed:

    VERDICT: [INCLUDE, UNDETERMINED, or FAIL]
    • INCLUDE: All inclusion criteria are definitively met based on available information
    • UNDETERMINED: Cannot determine eligibility due to missing critical information
    • FAIL: One or more inclusion criteria are definitively not met

    MISSING INFORMATION:
    [If UNDETERMINED: List specific information needed to make a determination]
    [If INCLUDE or FAIL: Write "None"]

    UNMET CRITERIA:
    [If FAIL: List which specific inclusion criteria are not met with exact criterion language]
    [If INCLUDE: Write "None"]
    [If UNDETERMINED: Write "Cannot determine" or list potentially unmet criteria]

    REASONING:
    [Provide a concise explanation of your decision, highlighting the key factors that led to your verdict]

    Your assessment must be evidence-based, referencing specific elements from both the trial criteria and patient summary.
    """
        }

    @staticmethod
    def _final_match_scoring():
        return {
            "system": "You are a senior oncology clinical trial investigator with expertise in patient selection and protocol design. You excel at comprehensive trial matching assessments that balance scientific rigor with clinical judgment. Your evaluations are evidence-based, nuanced, and focused on patient benefit.",
            "user": """
    Perform a comprehensive trial-patient match assessment, integrating previous filtering results and clinical judgment.

    Patient Profile:
    \"\"\"
    {patient_summary}
    \"\"\"

    Trial Description:
    \"\"\"
    {trial_summary}
    \"\"\"

    Previous Screening Results:
    • Exclusion Filter: {exclusion_verdict} - {exclusion_reason}
    • Inclusion Filter: {inclusion_verdict} - {inclusion_reasoning}
    • Missing Information: {missing_information}
    • Unmet Criteria: {unmet_criteria}

    ANALYSIS INSTRUCTIONS:
    Conduct a thorough evaluation of this match considering:
    1. Clinical appropriateness (is this the right trial for this patient?)
    2. Scientific eligibility (does the patient meet protocol requirements?)
    3. Practical feasibility (can the patient realistically participate?)
    4. Potential benefit (might this trial help this specific patient?)

    ASSESSMENT FRAMEWORK:
    • First, examine the alignment between patient diagnosis and trial target population
    • Next, evaluate biomarker/genetic compatibility if applicable
    • Assess treatment history alignment with protocol requirements
    • Consider performance status, organ function, and general eligibility
    • Review the previous screening results for any critical issues
    • Make a holistic determination considering all factors

    RATING SCALE:
    • 8-10: STRONG MATCH - Patient clearly meets all key criteria with strong alignment to trial target
    • 5-7: POSSIBLE MATCH - Patient likely meets most criteria but has some uncertainties to resolve
    • 0-4: UNSUITABLE - Patient fails to meet critical requirements or has multiple major uncertainties

    IMPORTANT: Your response MUST follow this EXACT FORMAT to be properly processed:

    SCORE: [Single numeric value 0-10]

    VERDICT: [HIGHLY LIKELY TO REFER / WOULD CONSIDER REFERRAL / WOULD NOT REFER]

    REASONING:
    [Provide a comprehensive assessment explaining your verdict, addressing:
    - Disease compatibility (type, stage, histology alignment)
    - Biomarker/genetic status compatibility
    - Treatment history alignment
    - Performance status and functional eligibility
    - Key eligibility factors influencing your decision
    - Any borderline or concerning issues]

    Focus on clinical significance rather than technical details, and prioritize the factors most relevant to this specific patient-trial match.
    """
        }

    @staticmethod
    def _patient_summary_sigir2016():
        return {
            "system": "You are a medical information specialist who excels at extracting structured clinical information from patient cases across all medical specialties. Your summaries are comprehensive, well-organized, and clinically precise.",
            "user": """
    Extract and organize the key medical information from this patient case.

    Patient Record:
    \"\"\"
    {patient_text}
    \"\"\"

    Create a structured summary with these EXACT sections:

    CHIEF COMPLAINT:
    • Main reason for visit/primary symptoms
    • Duration, severity, and pattern of symptoms
    • Associated symptoms or exacerbating factors

    VITAL SIGNS:
    • All measured vital parameters (temperature, blood pressure, heart rate, respiratory rate, etc.)
    • Any abnormal findings or trends
    • Relevant physical measurements

    KEY FINDINGS:
    • Important physical examination findings
    • Significant diagnostic test results (laboratory, imaging, etc.)
    • Abnormal findings requiring attention
    • Key clinical observations

    MEDICAL HISTORY:
    • Confirmed diagnoses and conditions
    • Previous surgeries, procedures, or hospitalizations
    • Risk factors (smoking status, alcohol use, exposures)
    • Family history if relevant to current presentation

    CURRENT MEDICATIONS:
    • All prescribed medications with dosages when available
    • Over-the-counter medications and supplements
    • Recent medication changes
    • Medication allergies or adverse reactions

    DEMOGRAPHICS:
    • Age, gender, and other relevant demographic information
    • Occupational factors if clinically relevant
    • Living situation if mentioned and relevant

    Use bullet points for clarity. Include ONLY information explicitly stated in the record. If information for a section is not provided, write "Not specified in record." Do not add inferred details not present in the original text.
    """
        }

    @staticmethod
    def _patient_condensed_sigir2016():
        return {
            "system": "You are a clinical data specialist who creates optimized patient representations for medical trial matching systems. You excel at identifying the most relevant clinical characteristics while standardizing information for semantic matching.",
            "user": """
    Create a CONDENSED PATIENT PROFILE (200-300 words) optimized for semantic matching with clinical trials.

    Patient Record:
    \"\"\"
    {patient_text}
    \"\"\"

    TASK: Generate a single coherent paragraph describing this patient's key clinical characteristics for trial matching. This text will be used for embedding-based similarity matching with trial descriptions.

    FOCUS EXCLUSIVELY ON:
    1. Primary medical condition(s) with specific diagnoses, severity, and staging
    2. Key symptomatology and clinical presentation
    3. Relevant diagnostic findings and test results
    4. Prior and current treatments
    5. Patient characteristics relevant to medical interventions
    6. Significant medical history affecting potential treatments

    CRITICAL GUIDELINES:
    • INCLUDE ONLY POSITIVE ATTRIBUTES that would qualify for clinical trials
    • CONVERT ALL NUMERIC VALUES to descriptive clinical terms:
      ✓ "adult patient" instead of "57 years old"
      ✓ "mild anemia" instead of "hemoglobin 9.2 g/dL"
      ✓ "near-normal kidney function" instead of "creatinine 1.2 mg/dL"
      ✓ "good functional status" instead of "ECOG 1"
    • USE PRECISE MEDICAL TERMINOLOGY consistently
    • WRITE AS A SINGLE COHERENT PARAGRAPH without subheadings
    • FOCUS ON CURRENT AND FACTUAL patient status
    • DO NOT include exclusion criteria or negative characteristics

    Your profile should enable optimal semantic similarity matching when compared with trial descriptions formatted in the same style.
    """
        }

    @staticmethod
    def _trial_summary_sigir2016():
        return {
            "system": "You are a clinical research expert specializing in protocol analysis across all medical specialties. You excel at extracting structured eligibility criteria from complex trial descriptions, making them clear and well-organized.",
            "user": """
    Create a structured summary of this clinical trial's key eligibility information.

    Trial Description:
    \"\"\"
    {trial_text}
    \"\"\"

    Extract and organize the information into these EXACT sections:

    TARGET CONDITION:
    • Primary condition or disease being studied
    • Specific disease subtypes, variants, or presentations included
    • Disease stage, severity, or classification requirements
    • Diagnostic criteria required for confirmation

    DEMOGRAPHICS:
    • Age range requirements (minimum/maximum)
    • Gender specifications if any
    • Other demographic requirements (BMI, weight, etc.)
    • Special population considerations

    INCLUSION REQUIREMENTS:
    • Essential symptoms, findings, or characteristics required
    • Necessary diagnostic test results or thresholds
    • Required medical history elements
    • Minimum duration or chronicity requirements
    • Other key qualifying factors

    EXCLUSION CRITERIA:
    • Disqualifying comorbidities or conditions
    • Prohibited medications, treatments, or interventions
    • Contraindicated patient characteristics
    • Other factors that would exclude patients

    INTERVENTION DETAILS:
    • Type of intervention (medication, procedure, device, etc.)
    • Dosing, frequency, or implementation specifics
    • Duration of intervention
    • Comparator or control group if mentioned

    Use bullet points to list each criterion clearly. If information for a section is not provided in the description, write "Not specified in protocol." Include only information explicitly stated in the trial description.
    """
        }

    @staticmethod
    def _trial_condensed_sigir2016():
        return {
            "system": "You are a clinical trial matching specialist who creates optimized trial representations for medical matching systems. You excel at identifying the most important eligibility factors while standardizing information for semantic retrieval.",
            "user": """
    Create a CONDENSED TRIAL PROFILE (200-300 words) optimized for semantic matching with patients.

    Trial Description:
    \"\"\"
    {trial_text}
    \"\"\"

    TASK: Generate a single coherent paragraph describing the ideal candidate for this clinical trial. This text will be used for embedding-based similarity matching with patient records.

    FOCUS EXCLUSIVELY ON:
    1. Target medical condition(s) with specific subtypes, severity, and staging
    2. Essential patient characteristics sought for enrollment
    3. Key qualifying clinical features or findings
    4. Prior treatment requirements or preferences
    5. Important physiological or functional parameters
    6. Unique eligibility factors specific to this trial

    CRITICAL GUIDELINES:
    • INCLUDE ONLY POSITIVE ATTRIBUTES that define eligible patients
    • CONVERT ALL NUMERIC THRESHOLDS to descriptive clinical terms:
      ✓ "adult patients" instead of "age > 18"
      ✓ "adequate hemoglobin levels" instead of "hemoglobin > 9 g/dL"
      ✓ "normal kidney function" instead of "creatinine < 1.5 mg/dL"
      ✓ "good functional status" instead of "ECOG 0-2"
    • DO NOT MENTION ANY EXCLUSION CRITERIA or negative characteristics
    • USE PRECISE MEDICAL TERMINOLOGY consistently
    • WRITE AS A SINGLE COHERENT PARAGRAPH without subheadings
    • Emphasize the most distinctive qualifying factors for this specific trial

    Your profile should enable optimal semantic similarity matching when compared with patient records formatted in the same style.
    """
        }