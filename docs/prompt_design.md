# Prompt Design for Semantic Trial Matching

## Core Principles

Effective prompt design is critical for semantic trial matching. This document outlines key principles we've discovered for optimizing embeddings in clinical trial matching systems.

## Parallel Prompt Structure

The most critical aspect of prompt design for embedding-based retrieval is ensuring **parallel structure** between trial and patient prompts. This significantly improves vector similarity matching.

### Why Parallel Structure Matters

When using embedding models and vector similarity (FAISS):

- Vector spaces align better when prompts request the same categories of information
- Similar language patterns lead to closer semantic representations
- Parallel structure ensures compatible abstraction levels
- Similar output formats produce more comparable embeddings

### Example: Parallel Trial and Patient Prompts

**Trial Prompt Structure:**
```
1. Target condition with specifics
2. Required patient characteristics
3. Required prior treatments
4. Essential inclusion criteria
5. Unique eligibility factors
```

**Patient Prompt Structure:**
```
1. Diagnosed conditions with specifics
2. Patient characteristics
3. Treatment history
4. Current disease status
5. Relevant medical information
```

## Avoiding Negation Problems

A key discovery was that **exclusion criteria should be omitted from embedding generation**, as vector similarity cannot understand negation properly.

### The Negation Problem

Vector similarity will match on terms regardless of negation context:

- Trial: "Exclusion: Patients with active cancer"  
- Patient: "Has stage III lung cancer"
- Result: High similarity on "cancer" despite being a disqualifying match!

### Solution

1. Remove exclusion criteria entirely from the trial condensed summary
2. Focus only on positive attributes in both trial and patient prompts
3. Reserve exclusion criteria for LLM-based filtering after retrieval
4. Use explicit instructions in prompts to avoid negative characteristics

## Numeric Value Normalization

Numeric values can cause false matches between unrelated information.

### The Numeric Matching Problem

Raw numbers in embeddings cause accidental similarities:

- Trial: "ALT must be < 57 U/L"
- Patient: "57-year-old female"
- Result: False match on "57"

### Solution

Convert numeric thresholds to clinical descriptors:

| Instead of | Use |
|------------|-----|
| "age > 18" | "adults" |
| "hemoglobin > 9 g/dL" | "adequate hemoglobin levels" |
| "creatinine < 1.5 mg/dL" | "normal kidney function" |
| "ECOG 0-1" | "good performance status" |
| "PD-L1 expression 80%" | "high PD-L1 expression" |

This preserves clinical meaning while eliminating numerical matching problems.

## Domain-Specific Optimization

Prompts should be tailored to the specific medical domain while following these principles.

### Oncology-Specific Considerations

For cancer trials, focus on:
- Cancer type, histology, and staging
- Biomarkers and genetic mutations
- Line of therapy and treatment history
- Performance status and organ function

### Other Medical Domains

When adapting to new domains:
1. Identify the key matching criteria specific to that field
2. Maintain parallel structure between trial and patient prompts
3. Convert domain-specific numeric values to descriptive terms
4. Exclude negative/exclusion information from embeddings

## Prompt Structure Guidelines

### Effective Prompt Components

1. **Clear system role** defining the expert perspective
2. **Explicit instruction** on the task and desired output format
3. **Specific focus areas** enumerated in parallel between trial and patient prompts
4. **Important instructions** about negation avoidance and numeric conversion
5. **Output formatting guidance** for consistency

### Example Instructions

```
IMPORTANT INSTRUCTIONS:
- Focus ONLY on positive attributes that would qualify for matching
- DO NOT include specific numeric thresholds - instead use descriptive terms
- DO NOT mention any exclusion criteria or negative characteristics
- Use domain-specific terminology consistent with clinical practice
```

## Impact on Retrieval Performance

Implementing these prompt design principles led to significant improvements:

- Capture rate of highly relevant trials increased from ~75% to >91%
- Reduction in false positives from numeric value confusion
- Better quality candidates for downstream LLM reasoning
- More consistent embedding representations

## Practical Implementation

When implementing new prompts:

1. Develop trial and patient prompts in parallel
2. Test with a representative sample of trials and patients
3. Analyze failed matches to identify prompt improvement opportunities
4. Iterate on prompt design based on capture rate metrics

The synergy between carefully designed prompts and embedding models is the foundation of effective semantic matching for clinical trials.