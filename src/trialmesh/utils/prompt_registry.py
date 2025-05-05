# src/trialmesh/utils/prompt_registry.py

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

            # New SIGIR2016-specific prompts
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
            "system": "You are a clinical assistant specializing in extracting structured patient information for clinical trial matching. Provide clear, concise, and medically accurate summaries.",
            "user": """
    Given the following patient medical history, organize the key information into CLEARLY LABELED SECTIONS.
    
    Patient Text:
    \"\"\"
    {patient_text}
    \"\"\"
    
    For each section, include all relevant information from the patient record:
    
    DIAGNOSIS:
    - List each cancer diagnosis with dates
    - Include stage, grade, and anatomical location
    
    BIOMARKERS:
    - List all genomic mutations (e.g., BRAF V600E, EGFR exon 19 deletion)
    - Include expression status (e.g., PD-L1 80%, HER2 3+)
    
    TREATMENT HISTORY:
    - List all cancer therapies with approximate dates
    - Include response and reason for discontinuation when available
    
    RELEVANT MEDICAL HISTORY:
    - Major comorbidities or conditions
    - Organ function issues
    - History that might affect trial eligibility
    
    CURRENT STATUS:
    - Performance status if mentioned
    - Current symptoms
    - Disease progression status
    
    Provide concise, bulleted information without adding details not present in the original text.
    """
        }

    @staticmethod
    def _patient_condensed():
        return {
            "system": "You are a clinical data specialist creating concise cancer patient summaries for AI-based trial matching systems. Focus on positive patient characteristics.",
            "user": """
    Generate a CONDENSED SUMMARY (UNDER 300 WORDS) of this cancer patient that contains ONLY information relevant for matching to appropriate clinical trials.

    Patient Record:
    \"\"\"
    {patient_text}
    \"\"\"

    Focus EXCLUSIVELY on:
    1. Primary and secondary cancer diagnoses with staging and histology
    2. Biomarkers and genetic mutations with expression levels
    3. Patient characteristics (gender, performance status)
    4. Prior cancer treatments with responses
    5. Current disease status and treatment needs
    6. Relevant organ function status affecting eligibility

    IMPORTANT INSTRUCTIONS:
    - Focus on POSITIVE attributes that would qualify the patient for trials
    - DO NOT include specific numeric thresholds - instead use descriptive terms
      * Instead of "52 years old" use "adult"
      * Instead of "PD-L1 expression 80%" use "high PD-L1 expression"
      * Instead of "creatinine 1.1 mg/dL" use "normal kidney function"
      * Instead of "ECOG 1" use "good performance status"
    - Include clinical significance rather than raw measurements
    - Focus on actionable cancer characteristics for trial matching

    Write as a continuous informative yet concise paragraph optimized for semantic embedding model matching with clinical trial criteria.
    """
        }

    @staticmethod
    def _trial_summary():
        return {
            "system": "You are a clinical research coordinator specializing in translating complex trial protocols into structured eligibility criteria.",
            "user": """
    Given the following trial description, organize the key information into CLEARLY LABELED SECTIONS to match patient profiles.
    
    Trial Text:
    \"\"\"
    {trial_text}
    \"\"\"
    
    Summarize the trial's target population and requirements:
    
    DIAGNOSIS:
    - Target cancer types
    - Required stage or disease status
    - Specific histology requirements
    
    BIOMARKERS:
    - Required biomarker status
    - Genetic mutations of interest
    - Expression level thresholds
    
    TREATMENT HISTORY:
    - Required prior therapies
    - Line of therapy requirements
    - Treatment-free interval requirements
    
    EXCLUSION CRITERIA:
    - Key medical conditions that disqualify patients
    - Prohibited concurrent medications
    - Organ function requirements
    
    INCLUSION CRITERIA:
    - Age and performance status requirements
    - Laboratory value thresholds
    - Other key eligibility factors
    
    Use clear, concise bullets focusing on objective criteria. Follow the exact section structure provided.
    """
        }

    @staticmethod
    def _trial_condensed():
        return {
            "system": "You are a trial eligibility specialist creating concise summaries for AI-based oncology matching systems. Focus exclusively on positive eligibility characteristics.",
            "user": """
    Generate a CONDENSED SUMMARY (UNDER 300 WORDS) of this clinical trial that focuses ONLY on what makes an ideal cancer patient match.

    Trial Description:
    \"\"\"
    {trial_text}
    \"\"\"

    Focus EXCLUSIVELY on:
    1. Target cancer type(s) with specific histology and staging requirements
    2. Required biomarkers and genetic mutations with expression levels
    3. Patient characteristics (gender, performance status)
    4. Prior cancer therapy requirements or preferences
    5. Essential inclusion criteria that qualify patients
    6. Unique eligibility factors specific to this oncology trial

    IMPORTANT INSTRUCTIONS:
    - DO NOT mention any exclusion criteria or negative characteristics
    - DO NOT include specific numeric thresholds - instead use descriptive terms
      * Instead of "age > 18" use "adults"
      * Instead of "ANC > 1.5 × 10⁹/L" use "adequate neutrophil count"
      * Instead of "creatinine < 1.5 mg/dL" use "normal kidney function"
      * Instead of "ECOG 0-1" use "good performance status"
    - Focus only on the positive attributes of ideal candidates

    Write as a continuous informative yet concise paragraph optimized for semantic embedding model matching with cancer patient records.
    """
        }

    @staticmethod
    def _exclusion_filter():
        return {
            "system": "You are a clinical trial screening specialist with expertise in evaluating exclusion criteria. Your task is to determine if a patient is definitively excluded from a trial based on the exclusion criteria. Be thorough in your assessment but only exclude when clear evidence exists.",
            "user": """
    Carefully review the trial's exclusion criteria against the patient's profile and determine if the patient should be excluded.

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
    3. Only mark as EXCLUDE if there is clear evidence the patient meets an exclusion criterion
    4. If information is missing to determine exclusion, give the patient the benefit of the doubt

    Return your analysis in this EXACT format:

    VERDICT: [EXCLUDE or PASS]

    REASON:
    [If EXCLUDE, explain specifically which exclusion criterion the patient meets and why]
    [If PASS, explain "Patient does not meet any exclusion criteria" or "Insufficient evidence to confirm any exclusion criteria"]
    """
        }

    @staticmethod
    def _inclusion_filter():
        return {
            "system": "You are a clinical trial eligibility specialist evaluating inclusion criteria. Your task is to determine if a patient is included to a trial based on the inclusion criteria. Be thorough in your assessment.",
            "user": """
    Compare the trial inclusion criteria to the patient's profile to determine if required inclusion conditions are met.
    
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
    3. Only mark as FAIL if there is clear evidence
    4. If information is missing to determine inclusion, give the patient the benefit of the doubt

    Return your analysis in this EXACT format:
    
    VERDICT: [INCLUDE, UNDETERMINED, or FAIL]
    - INCLUDE: All inclusion criteria are definitively met
    - UNDETERMINED: Cannot determine eligibility due to missing information
    - FAIL: One or more inclusion criteria are definitively not met
    
    MISSING INFORMATION:
    [If UNDETERMINED, list specific information needed to make a determination]
    [If INCLUDE or FAIL, write "None"]
    
    UNMET CRITERIA:
    [If FAIL, list which specific inclusion criteria are not met]
    [If INCLUDE or UNDETERMINED, write "None" or list potential concerns]
    
    REASONING:
    [Brief explanation of your decision, highlighting key factors]
    """
        }

    @staticmethod
    def _final_match_scoring():
        return {
            "system": "You are a clinical match evaluator with extensive experience in oncology trials. Provide balanced, evidence-based assessments of patient-trial compatibility focused on clinical significance rather than numerical values.",
            "user": """
    Carefully analyze how well the patient matches the trial requirements, considering previous filtering results, diagnosis alignment, biomarkers, prior treatments, and inclusion/exclusion criteria.

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
    - STRONG MATCH (Score 8-10): Patient CLEARLY meets ALL inclusion criteria and NO exclusion criteria apply. There is specific evidence for every major requirement. The patient's condition, biomarkers, and treatment history align perfectly with the trial's target population.

    - POSSIBLE MATCH (Score 5-7): Patient likely meets most key criteria, but has some uncertainties or missing information that requires verification. Core eligibility is still possible. Patient could benefit from the trial if questions are resolved.

    - UNSUITABLE (Score 0-4): Patient fails to meet one or more critical eligibility requirements OR has multiple uncertainties that make eligibility unlikely. Definitive contraindications may exist or there is insufficient evidence for key inclusion criteria.

    PROVIDE YOUR ASSESSMENT IN THIS EXACT FORMAT:

    SCORE: [0-10]

    VERDICT: [HIGHLY LIKELY TO REFER / WOULD CONSIDER REFERRAL / WOULD NOT REFER]

    REASONING: [Provide your conclusion emphasizing the factors that determined your verdict. Be specific about why the patient should or should not be referred to this trial]
    - explain any borderline concerns
    - Disease alignment (type, stage, histology)
    - Biomarker/genetic requirements
    - Prior treatment compatibility
    - Performance status and organ function eligibility

    Base your assessment on clinical significance rather than raw numerical values. Consider the totality of evidence and prioritize factors that would most impact the patient's eligibility and potential benefit.
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

