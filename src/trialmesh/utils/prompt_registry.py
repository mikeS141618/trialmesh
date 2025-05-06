# src/trialmesh/utils/prompt_registry.py

"""
PromptRegistry for Semantic Clinical Trial Matching

This registry provides parallel, domain-optimized prompt templates for
embedding-based retrieval and LLM-based filtering in clinical trial matching.
Prompts are designed for maximal compatibility with Llama 3.3 70B and similar
LLMs, and encode empirically validated best practices:

- Parallel structure between patient and trial prompts for embedding alignment
- Exclusion of negation and exclusion criteria from embedding prompts
- Numeric value normalization to clinical descriptors
- Explicit output formatting and section ordering
- Domain-specific variants (e.g., oncology, SIGIR2016)
- Deterministic, concise, and role-focused system prompts

All templates are versioned and documented for maintainability.
"""

from typing import Dict

class PromptRegistry:
    """
    Registry of prompt templates for LLM interactions in trial matching.

    Each template is a dict with 'system' and 'user' keys.
    Prompts are parallelized for embedding-based retrieval, and
    include explicit output formatting and normalization instructions.
    """

    def __init__(self):
        self.prompts: Dict[str, Dict[str, str]] = {
            # Embedding prompts (parallel structure, no negation, normalized numerics)
            "patient_condensed": self._patient_condensed(),
            "trial_condensed": self._trial_condensed(),
            "patient_condensed_sigir2016": self._patient_condensed_sigir2016(),
            "trial_condensed_sigir2016": self._trial_condensed_sigir2016(),

            # Structured summary prompts (for downstream LLM reasoning)
            "patient_summary": self._patient_summary(),
            "trial_summary": self._trial_summary(),
            "patient_summary_sigir2016": self._patient_summary_sigir2016(),
            "trial_summary_sigir2016": self._trial_summary_sigir2016(),

            # LLM-based filtering and scoring (may include negation)
            "exclusion_filter": self._exclusion_filter(),
            "inclusion_filter": self._inclusion_filter(),
            "final_match_scoring": self._final_match_scoring(),
        }

    def get(self, name: str) -> Dict[str, str]:
        """Get prompt pair (system and user) by name."""
        return self.prompts.get(name, {"system": "", "user": ""})

    def get_system(self, name: str) -> str:
        """Get just the system prompt for a template."""
        return self.prompts.get(name, {}).get("system", "")

    def get_user(self, name: str) -> str:
        """Get just the user prompt for a template."""
        return self.prompts.get(name, {}).get("user", "")

    # === Embedding Prompts (Parallel, No Negation, Normalized Numerics) ===

    @staticmethod
    def _patient_condensed():
        return {
            "system": "You are a clinical data specialist.",
            "user": """
Summarize the following cancer patient record in under 300 words, focusing ONLY on positive attributes relevant for clinical trial matching. Use a single concise paragraph.

Patient Record:
\"\"\"
{patient_text}
\"\"\"

Focus exclusively on:
1. Primary and secondary cancer diagnoses (with staging and histology)
2. Biomarkers and genetic mutations (with expression levels)
3. Patient characteristics (gender, performance status)
4. Prior cancer treatments and responses
5. Current disease status and treatment needs
6. Relevant organ function status affecting eligibility

IMPORTANT INSTRUCTIONS:
- DO NOT include any exclusion criteria or negative characteristics
- DO NOT include specific numeric thresholds; use descriptive clinical terms (e.g., "adult", "high PD-L1 expression", "good performance status")
- Use only information present in the text; do not add or infer details
- Use domain-specific terminology consistent with clinical practice

Write a single informative paragraph. Do not use section headings or lists.
"""
        }

    @staticmethod
    def _trial_condensed():
        return {
            "system": "You are a clinical trial eligibility specialist.",
            "user": """
Summarize the following clinical trial in under 300 words, focusing ONLY on positive eligibility attributes for ideal cancer patient matches. Use a single concise paragraph.

Trial Description:
\"\"\"
{trial_text}
\"\"\"

Focus exclusively on:
1. Target cancer type(s) (with histology and staging)
2. Required biomarkers and genetic mutations (with expression levels)
3. Patient characteristics (gender, performance status)
4. Prior cancer therapy requirements or preferences
5. Essential inclusion criteria that qualify patients
6. Unique eligibility factors specific to this trial

IMPORTANT INSTRUCTIONS:
- DO NOT mention any exclusion criteria or negative characteristics
- DO NOT include specific numeric thresholds; use descriptive clinical terms (e.g., "adults", "adequate neutrophil count", "good performance status")
- Use only information present in the text; do not add or infer details
- Use domain-specific terminology consistent with clinical practice

Write a single informative paragraph. Do not use section headings or lists.
"""
        }

    @staticmethod
    def _patient_condensed_sigir2016():
        return {
            "system": "You are a clinical data specialist.",
            "user": """
Summarize the following patient record in under 300 words, focusing ONLY on positive attributes relevant for clinical trial matching. Use a single concise paragraph.

Patient Record:
\"\"\"
{patient_text}
\"\"\"

Focus exclusively on:
1. Primary medical condition(s) (with subtypes and staging)
2. Patient characteristics (gender, performance status)
3. Relevant clinical findings and diagnostic results
4. Prior treatments and current therapy status
5. Significant medical history affecting eligibility

IMPORTANT INSTRUCTIONS:
- DO NOT include any exclusion criteria or negative characteristics
- DO NOT include specific numeric thresholds; use descriptive clinical terms (e.g., "adult", "mild anemia", "good performance status")
- Use only information present in the text; do not add or infer details
- Use domain-specific terminology consistent with clinical practice

Write a single informative paragraph. Do not use section headings or lists.
"""
        }

    @staticmethod
    def _trial_condensed_sigir2016():
        return {
            "system": "You are a clinical trial eligibility specialist.",
            "user": """
Summarize the following clinical trial in under 300 words, focusing ONLY on positive eligibility attributes for ideal patient matches. Use a single concise paragraph.

Trial Description:
\"\"\"
{trial_text}
\"\"\"

Focus exclusively on:
1. Target medical condition(s) (with subtypes and staging)
2. Required patient characteristics (gender, performance status)
3. Essential inclusion criteria for most participants
4. Prior treatments that are required or preferred
5. Unique eligibility factors specific to this trial

IMPORTANT INSTRUCTIONS:
- DO NOT mention any exclusion criteria or negative characteristics
- DO NOT include specific numeric thresholds; use descriptive clinical terms (e.g., "adults", "adequate hemoglobin levels", "good performance status")
- Use only information present in the text; do not add or infer details
- Use domain-specific terminology consistent with clinical practice

Write a single informative paragraph. Do not use section headings or lists.
"""
        }

    # === Structured Summaries (for LLM Reasoning) ===

    @staticmethod
    def _patient_summary():
        return {
            "system": "You are a clinical assistant. Your summaries are clear, concise, and medically accurate.",
            "user": """
Given the following patient medical history, extract and organize the information into the following sections, using the exact section headings below. Use concise bullet points. Do not add any information not present in the text.

Patient Text:
\"\"\"
{patient_text}
\"\"\"

Output format (use these exact headings):

# DIAGNOSIS
- [List each cancer diagnosis with dates, stage, grade, and anatomical location]

# BIOMARKERS
- [List all genomic mutations and expression status]

# TREATMENT HISTORY
- [List all cancer therapies with approximate dates, response, and reason for discontinuation]

# RELEVANT MEDICAL HISTORY
- [Major comorbidities, organ function issues, relevant history]

# CURRENT STATUS
- [Performance status, current symptoms, disease progression]

If a section has no information, write "None".

Do not include any details not present in the original text. Output must use the exact section headings above.
"""
        }

    @staticmethod
    def _trial_summary():
        return {
            "system": "You are a clinical research coordinator. Your summaries are structured and clinically precise.",
            "user": """
Given the following trial description, extract and organize the information into the following sections, using the exact section headings below. Use concise bullet points. Do not add any information not present in the text.

Trial Text:
\"\"\"
{trial_text}
\"\"\"

Output format (use these exact headings):

# DIAGNOSIS
- [Target cancer types, required stage/status, histology]

# BIOMARKERS
- [Required biomarker status, genetic mutations, expression thresholds]

# TREATMENT HISTORY
- [Required prior therapies, line of therapy, treatment-free interval]

# INCLUSION CRITERIA
- [Age, performance status, lab value thresholds, other key eligibility factors]

# EXCLUSION CRITERIA
- [Key medical conditions, prohibited medications, organ function requirements]

If a section has no information, write "None".

Do not include any details not present in the original text. Output must use the exact section headings above.
"""
        }

    @staticmethod
    def _patient_summary_sigir2016():
        return {
            "system": "You are a clinical assistant. Your summaries are clear, concise, and medically accurate.",
            "user": """
Given the following patient medical history, extract and organize the information into the following sections, using the exact section headings below. Use concise bullet points. Do not add any information not present in the text.

Patient Text:
\"\"\"
{patient_text}
\"\"\"

Output format (use these exact headings):

# CHIEF COMPLAINT
- [Main reason for visit/primary symptoms, duration, severity]

# VITAL SIGNS
- [Measured vitals]

# KEY FINDINGS
- [Physical exam, labs, imaging]

# MEDICAL HISTORY
- [Existing conditions, surgeries, risk factors]

# CURRENT MEDICATIONS
- [Prescribed and OTC medications]

# DEMOGRAPHICS
- [Age, gender, other relevant info]

If a section has no information, write "None".

Do not include any details not present in the original text. Output must use the exact section headings above.
"""
        }

    @staticmethod
    def _trial_summary_sigir2016():
        return {
            "system": "You are a clinical research coordinator. Your summaries are structured and clinically precise.",
            "user": """
Given the following trial description, extract and organize the information into the following sections, using the exact section headings below. Use concise bullet points. Do not add any information not present in the text.

Trial Text:
\"\"\"
{trial_text}
\"\"\"

Output format (use these exact headings):

# TARGET CONDITION
- [Primary condition, stage/severity requirements]

# DEMOGRAPHICS
- [Age, gender, other criteria]

# INCLUSION REQUIREMENTS
- [Symptoms, diagnostic criteria, lab/test thresholds, medical history]

# EXCLUSION CRITERIA
- [Disqualifying comorbidities, prohibited medications, other exclusion factors]

# INTERVENTION DETAILS
- [Intervention, dosing, frequency, administration]

If a section has no information, write "None".

Do not include any details not present in the original text. Output must use the exact section headings above.
"""
        }

    # === LLM-Based Filtering and Scoring Prompts ===

    @staticmethod
    def _exclusion_filter():
        return {
            "system": "You are a clinical trial screening specialist.",
            "user": """
Review the trial's exclusion criteria and the patient's summary. Determine if the patient should be excluded. Only exclude if there is clear evidence. If information is missing, give the patient the benefit of the doubt.

Trial Exclusion Criteria:
\"\"\"
{exclusion_criteria}
\"\"\"

Patient Summary:
\"\"\"
{patient_summary}
\"\"\"

Output format (use this exact structure):

VERDICT: [EXCLUDE or PASS]

REASON:
[If EXCLUDE, specify which exclusion criterion is met and why. If PASS, state "Patient does not meet any exclusion criteria" or "Insufficient evidence to confirm any exclusion criteria".]

Do not include any information not present in the input. Use the exact output format above.
"""
        }

    @staticmethod
    def _inclusion_filter():
        return {
            "system": "You are a clinical trial eligibility specialist.",
            "user": """
Compare the trial inclusion criteria to the patient's summary to determine if required inclusion conditions are met.

Trial Inclusion Criteria:
\"\"\"
{inclusion_criteria}
\"\"\"

Patient Summary:
\"\"\"
{patient_summary}
\"\"\"

Output format (use this exact structure):

VERDICT: [INCLUDE, UNDETERMINED, or FAIL]
- INCLUDE: All inclusion criteria are definitively met
- UNDETERMINED: Cannot determine eligibility due to missing information
- FAIL: One or more inclusion criteria are definitively not met

MISSING INFORMATION:
[List specific information needed if UNDETERMINED, otherwise "None"]

UNMET CRITERIA:
[List which inclusion criteria are not met if FAIL, otherwise "None"]

REASONING:
[Brief explanation of your decision, highlighting key factors]

Do not include any information not present in the input. Use the exact output format above.
"""
        }

    @staticmethod
    def _final_match_scoring():
        return {
            "system": "You are a clinical match evaluator with expertise in clinical trials.",
            "user": """
Analyze how well the patient matches the trial requirements, considering previous filtering results, diagnosis alignment, biomarkers, prior treatments, and inclusion/exclusion criteria.

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

Classification guidelines:
- STRONG MATCH (Score 8-10): Patient clearly meets all inclusion criteria and no exclusion criteria apply. All major requirements are supported by evidence.
- POSSIBLE MATCH (Score 5-7): Patient likely meets most key criteria, but some uncertainties or missing information remain.
- UNSUITABLE (Score 0-4): Patient fails to meet one or more critical requirements or has multiple uncertainties.

Output format (use this exact structure):

SCORE: [0-10]

VERDICT: [HIGHLY LIKELY TO REFER / WOULD CONSIDER REFERRAL / WOULD NOT REFER]

REASONING:
[Provide your conclusion, emphasizing the factors that determined your verdict. Be specific about disease alignment, biomarkers, prior treatment compatibility, and performance status.]

Base your assessment on clinical significance rather than raw numerical values. Use only information present in the input.
"""
        }