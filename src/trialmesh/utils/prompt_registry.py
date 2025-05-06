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
            "exclusion_filter_sigir2016": self._exclusion_filter_sigir2016(),
            "inclusion_filter_sigir2016": self._inclusion_filter_sigir2016(),
            "final_match_scoring_sigir2016": self._final_match_scoring_sigir2016(),
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
            "system": "You are a clinical assistant specializing in extracting structured information from patient cases across all medical specialties. Provide clear, concise, and medically accurate summaries.",
            "user": """
    Given the following patient medical history, organize the key information into CLEARLY LABELED SECTIONS.

    Patient Text:
    \"\"\"
    {patient_text}
    \"\"\"

    For each section, include all relevant information from the patient record:

    CHIEF COMPLAINT:
    - Main reason for visit/primary symptoms
    - Duration and severity

    VITAL SIGNS:
    - Any measured vitals (temperature, blood pressure, heart rate, respiratory rate, etc.)

    KEY FINDINGS:
    - Significant physical examination findings
    - Relevant laboratory or imaging results

    MEDICAL HISTORY:
    - Existing conditions
    - Previous surgeries or hospitalizations
    - Risk factors (smoking, travel history, exposures)

    CURRENT MEDICATIONS:
    - Prescribed medications
    - Over-the-counter medications

    DEMOGRAPHICS:
    - Age, gender, and other relevant demographic information

    Provide concise, bulleted information without adding details not present in the original text.
    """
        }

    @staticmethod
    def _patient_condensed_sigir2016():
        return {
            "system": "You are a clinical data specialist creating concise patient summaries for AI-based trial matching systems. Focus on positive patient characteristics.",
            "user": """
    Generate a CONDENSED SUMMARY (UNDER 300 WORDS) of this patient that contains ONLY information relevant for matching to appropriate clinical trials.

    Patient Record:
    \"\"\"
    {patient_text}
    \"\"\"

    Focus EXCLUSIVELY on:
    1. Primary medical condition(s) with specific subtypes and staging
    2. Patient characteristics (gender, performance status)
    3. Relevant clinical findings and diagnostic results
    4. Prior treatments received and current therapy status
    5. Significant medical history that affects eligibility for interventions

    IMPORTANT INSTRUCTIONS:
    - Focus on POSITIVE attributes that would qualify the patient for trials
    - DO NOT include specific numeric thresholds - instead use descriptive terms
      * Instead of "57 years old" use "adult"
      * Instead of "hemoglobin 9.2 g/dL" use "mild anemia" or "slightly low hemoglobin"
      * Instead of "creatinine 1.2 mg/dL" use "near-normal kidney function"
      * Instead of "ECOG 1" use "good performance status"
    - Include clinical significance rather than raw measurements

    Write as a continuous informative yet concise paragraph optimized for semantic embedding model matching with clinical trial criteria.
    """
        }

    @staticmethod
    def _trial_summary_sigir2016():
        return {
            "system": "You are a clinical research coordinator specializing in translating complex trial protocols into structured eligibility criteria for all medical specialties.",
            "user": """
    Given the following trial description, organize the key information into CLEARLY LABELED SECTIONS to match potential participants.

    Trial Text:
    \"\"\"
    {trial_text}
    \"\"\"

    Summarize the trial's target population and requirements:

    TARGET CONDITION:
    - Primary condition being studied
    - Disease stage or severity requirements

    DEMOGRAPHICS:
    - Age range requirements
    - Gender specifications
    - Other demographic criteria

    INCLUSION REQUIREMENTS:
    - Key symptoms or diagnostic criteria
    - Laboratory or test result thresholds
    - Specific medical history requirements

    EXCLUSION CRITERIA:
    - Disqualifying comorbidities
    - Prohibited medications or therapies
    - Other factors that would exclude patients

    INTERVENTION DETAILS:
    - Brief description of the intervention
    - Dosing, frequency, or administration details if relevant

    Use clear, concise bullets focusing on objective criteria. Follow the exact section structure provided.
    """
        }

    @staticmethod
    def _trial_condensed_sigir2016():
        return {
            "system": "You are a clinical trial eligibility specialist. Focus exclusively on the positive eligibility characteristics.",
            "user": """
    Generate a CONDENSED SUMMARY (UNDER 300 WORDS) of this clinical trial that focuses ONLY on what makes an ideal candidate match.

    Trial Description:
    \"\"\"
    {trial_text}
    \"\"\"

    Focus EXCLUSIVELY on:
    1. Target medical condition(s) with specific subtypes and staging
    2. Required patient characteristics (gender, performance status)
    3. Essential inclusion criteria that most participants must meet
    4. Prior treatments that are required or preferred
    5. Any unique eligibility factors specific to this trial

    IMPORTANT INSTRUCTIONS:
    - DO NOT mention any exclusion criteria or negative characteristics
    - DO NOT include specific numeric thresholds - instead use descriptive terms
      * Instead of "age > 18" use "adults"
      * Instead of "hemoglobin > 9 g/dL" use "adequate hemoglobin levels"
      * Instead of "creatinine < 1.5 mg/dL" use "normal kidney function"
      * Instead of "ECOG 0-2" use "good performance status"
    - Focus only on the positive attributes of ideal candidates

    Write as a continuous informative yet concise paragraph optimized for semantic embedding model matching with patient records.
    """
        }

    @staticmethod
    def _exclusion_filter_sigir2016():
        return {
            "system": "You are a medical research coordinator evaluating patient eligibility for clinical trials. You understand that clinical trials often present valuable treatment options, and you aim to give patients the benefit of the doubt whenever reasonable.",
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

    INSTRUCTIONS:
    1. Review each exclusion criterion carefully
    2. Compare against the patient's information
    3. Only mark as EXCLUDE if there is strong, clear evidence the patient meets an exclusion criterion
    4. Give the patient the benefit of the doubt when information is missing or ambiguous
    5. Consider that many exclusion criteria have exceptions or can be waived in appropriate cases

    Return your analysis in this EXACT format:

    VERDICT: [EXCLUDE or PASS]

    REASON:
    [If EXCLUDE, explain specifically which exclusion criterion the patient meets and why]
    [If PASS, explain "Patient does not meet any exclusion criteria" or "Insufficient evidence to confirm any exclusion criteria"]
    """
        }

    @staticmethod
    def _inclusion_filter_sigir2016():
        return {
            "system": "You are a medical research coordinator evaluating patient eligibility for clinical trials. You understand that clinical trials often present valuable treatment options, and you aim to give patients the benefit of the doubt whenever reasonable.",
            "user": """
    Evaluate whether this patient meets the inclusion criteria for the clinical trial.

    Trial Inclusion Criteria:
    \"\"\"
    {inclusion_criteria}
    \"\"\"

    Patient Summary:
    \"\"\"
    {patient_summary}
    \"\"\"

    INSTRUCTIONS:
    1. Review each inclusion criterion carefully
    2. Compare against the patient's information
    3. Only mark as FAIL if there is strong, clear evidence that a criterion is not met
    4. Give the patient the benefit of the doubt when information is missing or ambiguous
    5. Consider that many inclusion criteria have flexibility or can be interpreted favorably when appropriate

    Return your analysis in this EXACT format:

    VERDICT: [INCLUDE, UNDETERMINED, or FAIL]
    - INCLUDE: All inclusion criteria appear to be met based on available information
    - UNDETERMINED: Cannot determine eligibility due to missing key information
    - FAIL: One or more inclusion criteria are clearly not met

    MISSING INFORMATION:
    [If UNDETERMINED, list specific information needed to make a determination]
    [If INCLUDE or FAIL, write "None"]

    UNMET CRITERIA:
    [If FAIL, list which specific inclusion criteria are not met]
    [If INCLUDE or UNDETERMINED, write "None" or list potential minor concerns]

    REASONING:
    [Brief explanation of your decision, highlighting key factors]
    """
        }

    @staticmethod
    def _final_match_scoring_sigir2016():
        return {
            "system": "You are a medical research coordinator evaluating patient-trial matches. You understand the importance of connecting patients with appropriate clinical trials, and you aim to identify all reasonable opportunities for patient participation.",
            "user": """
    Evaluate how well this patient matches this clinical trial, considering all available information.

    Patient Profile:
    \"\"\"
    {patient_summary}
    \"\"\"

    Trial Description:
    \"\"\"
    {trial_summary}
    \"\"\"

    PREVIOUS EVALUATION RESULTS:
    Exclusion Filter: {exclusion_verdict} - {exclusion_reason}
    Inclusion Filter: {inclusion_verdict} - {inclusion_reasoning}
    Missing Information: {missing_information}
    Unmet Criteria: {unmet_criteria}

    CLASSIFICATION GUIDELINES:
    - STRONG MATCH (Score 7-10): Patient meets all major eligibility criteria and would likely benefit from this trial. Any minor questions or issues could reasonably be addressed or waived.

    - POSSIBLE MATCH (Score 4-6): Patient meets some key criteria but has uncertainties or missing information that need verification. The patient could be eligible if these questions are resolved favorably.

    - UNSUITABLE (Score 0-3): Patient clearly fails to meet multiple critical eligibility requirements or has a condition that would make trial participation inappropriate or unsafe.

    PROVIDE YOUR ASSESSMENT IN THIS EXACT FORMAT:

    SCORE: [0-10]

    VERDICT: [HIGHLY LIKELY TO REFER / WOULD CONSIDER REFERRAL / WOULD NOT REFER]

    REASONING: [Provide your conclusion emphasizing why the patient should or should not be referred to this trial]
    - Medical condition alignment with trial focus
    - Key eligibility criteria met or not met
    - Potential benefits for this specific patient
    - Any important concerns or uncertainties

    Remember that clinical trials often provide valuable options for patients, especially when standard treatments have limitations. When in doubt about borderline cases, consider the potential value to the patient.
    """
        }