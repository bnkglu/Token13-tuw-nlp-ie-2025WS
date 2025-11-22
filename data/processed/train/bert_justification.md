
# Project Component: Machine Learning Baseline & Integration Strategy
**Topic 8: Explainable Relation Extraction**

## 1. Executive Summary
[cite_start]This document outlines the architectural decision to utilize a Transformer-based architecture (RoBERTa/BERT) as the Machine Learning baseline for Milestone 2. Furthermore, it defines the strategic roadmap for Milestone 3, specifically addressing how this "black box" model will be leveraged to generate explainable patterns for the rule-based system, satisfying the requirement to "leverage machine learning models for creating patterns"[cite: 247].

## 2. Milestone 2: The Machine Learning Baseline

[cite_start]For the quantitative baseline required by Milestone 2[cite: 15], we have implemented a fine-tuned **RoBERTa-base** model.

### 2.1 Rationale for Transformer Architecture
While traditional statistical models (SVM, Logistic Regression) were considered, RoBERTa was selected for the following reasons:

* **Handling Contextual Dependency:** Relation Extraction often relies on long-distance dependencies between entities (e.g., *Entity A*... [10 words] ... *caused* ... *Entity B*). Transformer attention mechanisms capture these dependencies far more effectively than window-based or bag-of-words approaches.
* **Entity Marker Compatibility:** Our preprocessing strategy utilizes special tokens (e.g., `[E1]`, `[/E1]`) to demarcate entities. BERT-based models excel at learning relative positioning and semantic roles from these markers without requiring manual feature engineering (dependency tree parsing depth, POS tagging, etc.).
* **Establishing a Performance Ceiling:** To evaluate the quality of our final "Explainable Solution," we must first establish the "Performance Ceiling"—the maximum achievable F1-score using current State-of-the-Art (SOTA) methods. RoBERTa provides this rigorous benchmark.

### 2.2 Implementation Details
* **Input Format:** Raw text with inserted entity markers.
* **Model:** `roberta-base` (12-layer, 768-hidden, 12-heads).
* **Objective:** Multi-class classification (classifying the relation type between marked entities).

---

## 3. Milestone 3: Integration & Convergence Strategy

[cite_start]The core challenge of Topic 8 is balancing **performance** with **explainability**[cite: 246]. Our strategy is not to treat the ML model and Rule-based model as competitors, but to use the ML model as a **Pattern Discovery Tool** for the rule-based system.

### 3.1 The "Teacher-Student" Pattern Mining Workflow
We will employ a semi-supervised "distillation" approach to leverage the ML model's high recall to improve the Rule-based system's precision.

#### Phase 1: High-Confidence Filtering (The "Scout")
Rule-based systems (like POTATO) require high-quality examples to extract valid graph patterns. Manually searching for these examples is inefficient.
1.  We will deploy the trained RoBERTa model on the test set (and potentially unlabeled external data).
2.  We will filter the predictions to retain only **High-Confidence Instances** (e.g., Softmax probability $> 0.95$).
3.  **Hypothesis:** High model confidence correlates with strong, explicit linguistic signals (e.g., specific trigger words or clear syntactic structures).

#### Phase 2: Pattern Extraction (The "Analyst")
These high-confidence examples will serve as the "Gold Standard" input for the rule-based component (POTATO).
* **Input:** The 500+ high-confidence sentences identified by RoBERTa.
* **Process:** POTATO will analyze the dependency graphs of these specific sentences to identify common sub-graphs.
* **Output:** A set of human-interpretable rules (e.g., `(Subject) --nsubj--> (active_verb) --dobj--> (Object)`).

### 3.2 The Hybrid Triage System (Final Architecture)
[cite_start]For the final solution submission in January[cite: 20], we propose a hybrid inference pipeline:

| Stage | Component | Function | Goal |
| :--- | :--- | :--- | :--- |
| **1** | **Rule System** | Checks input against strict, explainable patterns. | **High Precision / Explainability.** If a rule triggers, we accept the answer and provide the rule as the explanation. |
| **2** | **ML Fallback** | If no rule triggers, the RoBERTa model predicts the relation. | **High Recall.** Captures subtle or implicit relations that rules miss, ensuring the system is still robust. |

## 4. Conclusion
This approach satisfies the course requirements by:
1.  [cite_start]Providing a strong **Machine Learning Baseline** (Milestone 2)[cite: 16].
2.  [cite_start]Ensuring the final solution focuses on **Explainability** (Topic 8 Goal)[cite: 243].
3.  [cite_start]Directly **leveraging ML** to assist in the creation of rules/patterns[cite: 247].