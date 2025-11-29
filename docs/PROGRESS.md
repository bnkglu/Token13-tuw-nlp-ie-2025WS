# Progress Log - Explainable Relation Extraction (Topic 8)

[Google Docs for Progress and more details](https://docs.google.com/document/d/1z_I0ro6DkJUpFkJBYeFXhau7JllqXYldTa7aKoK6NNg/edit?usp=sharing)

(Ensure the document's sharing settings permit access for team members.)

## 22 Oct 2025
- GitHub repo created.

## 26 Oct 2025 - Initial Meeting
- Dataset was selected: SemEval 2010 Task 8 (source: https://semeval2.fbk.eu/Data.html).

- Until next meeting: All plan preprocessing (cleaning, tokenization, extraction of given entities/relation) by 29 Oct and divide tasks.
- Considered: SemEval 2026 (feasibility TBD).
- Send an email to professor for dataset approval.

## 01.11.2025 - Final Meeting before Milestone 1
- Decided on which implementation to use for Milestone 1.

- Discussed plans for Milestone 2:
    1. Everyone will research baseline models can be implemented for relation extraction. After that, we will divide the models to implement.
        - What Relation Extraction does?
        - How can be these models implemented?
            - What are the key steps?
                - What are the inputs of these models?
            - Can we use conll-u format for these models directly or do we need to add more features?
        - Can they be improved?
        - Are there some examples of these models being used for relation extraction?
        - What are the pros and cons of these models?
        - Are there any research papers about our topic and these models?
- **Milestone 2: meeting is on 14.11.2025 14:00-15:30.**


**Note from Lecture Text Classification:**
- For Milestone 2 of the Project exercise you should perform error analysis on your baseline models and discuss your findings, including implications for how your solution could be improved.


## 14.11.2025 - Milestone 2 Meeting
- Next meeting 24.11.2025
- More in detail research about models to implement.
- 2 Person(Bilal & EgeO) will implement ML models. (e.g. Logistic Regression, SVM, etc.)
- 2 Person(Berke & EgeA) will implement Rule-Based models. (e.g. POTATO, Pattern Matching, Dependency Parsing, etc.)
- Review literature for more models.
- At the next meeting, we will check results, problems faced, and next steps.

- Basic steps:
    1- Research on literature for SemEval, models and different approaches.
    2- Compare findings and choose a model to implement state the reason.
    3- Implement the chosen model and evaluate its performance.
    4- Analyze errors, challenges and discuss potential improvements.
    5- Document the process and findings for Milestone 2 report.

## 24.11.2025 - Progress Meeting
    - Each pair presented their progress.
    - Discussed challenges faced during implementation.

    - Berke:
        - Try POTATO with full data and gold data of RoBERTa.
        - Rule-based approach concise and interpretable.
        - will add interpretation, and explanation methods to Rule-Based models.
        - **Qualitative and Quantitative explanation of results.**
    - Bilal:
        - will add interpretation, and explanation methods to ML models.
        - one DL implementation. (make literature research for this implementations to see what can be done)
        - **Qualitative and Quantitative explanation of results.**
    - EgeA:
        - Rule-Based implementation
        - Interpretation of the results
        - **Qualitative and Quantitative explanation of results.**
    - EgeO 
        - will make your implementation in both way. 
        - how can we use this implementation with Rule-Based systems?
        - **Qualitative and Quantitative explanation of results.**

    - Next Meeting TBA: