# Token13-tuw-nlp-ie-2025WS

This repository belongs to the Token13 group, created for tuw-nlp-ie-2025WS's project.

# Topic 8 - Explainable Relation Extraction
**Instructors** Gábor Recski

## Overview 
Relation extraction is the task of finding pairs of entities in text that are in one of a few predefined semantic relationships. RE is a common task in industry NLP applications, but the so-called “state-of-the-art” models often cannot be deployed due to their lack of configurability and predictability. The goal of this task is to develop rule systems for relation extraction by leveraging machine learning models for creating patterns that can then be applied to input text. Some helpful tools for building rule-based systems can be the [spaCy functionality](https://spacy.io/api/dependencymatcher) for building patterns on dependency trees or the [POTATO](https://github.com/adaamko/POTATO) library for extracting and crafting graph patterns for text classification. But building a rule-based system from scratch or based on the code of any existing open-source system is also fine.

## Milestone Descriptions

Milestone 1 By November 2 you will have your core text datasets preprocessed and stored in a standard format. More detail will be provided in the lecture on text processing (Week 2). Topic descriptions will contain suggestions for possible datasets.

Milestone 2 By November 30 you will have implemented multiple baseline solutions to your main text classification task. These should include both machine learning methods and rule-based methods. Each baseline should be evaluated both quantitatively and qualitatively. More details will be provided in the lecture on text classification (Week 3). Topic descriptions may contain additional details about possible baseline approaches.

Final solution Your final solution is due by the end of January 25. Final presentations will take place on January 16 (a week after the final lecture), the week after that should be reserved for improvements based on feedback from the presentation. Your final submission should include all your code with documentation, a management summary (see Section 2), and your presentation slides.

## Instructions

## Instructions

**Evaluation**  
Proper evaluation of methods, including your own, both quantitative (e.g. precision and recall) and qualitative (e.g. looking at the data), is essential. For some tasks and some datasets you cannot assume that higher figures mean better solutions. Some manual analysis of a system’s output is usually necessary to understand its strengths and limitations. Topic descriptions may indicate task-specific challenges of evaluation.

**Technical details**  
Teams should create a repository on GitHub, add their mentor as a collaborator, and push their solutions to this repository. Your solution should be implemented in Python 3.7 or higher and should generally conform to PEP8 guidelines. You should also observe clean code principles. Teams should use the repository for version control and collaboration, as opposed to pushing their solutions in bulk before the deadline. Your codebase should be reasonably well organized. Submitting your solutions as large jupyter notebooks is therefore highly discouraged. Extensive documentation of the codebase is not necessary, but the README should describe the high-level structure and provide clear and concise instructions on how key results should be reproduced.

**Management summary**  
Your submission must be accompanied by a 2-page PDF document that presents a summary of your solution — this is a management summary, so it should be written in a way that is easy to understand by non-technical stakeholders, not NLP colleagues. The summary should contain an overview of the task, the challenges you faced, the external resources you used, the solution you implemented and its limitations, and possible next steps. It should also briefly describe the contributions of each of the team members, pointing out any unforeseen issues (e.g. a team member dropping out or contributing significantly less than what was agreed upon).

**Final Presentation**  
Each group will present the main results of their work to all other groups working on the same topic. The format is 15 minutes of presentation and 5 minutes of discussion — we will be very strict with the timing, and stop the presentation at the 15 minute mark. Each team member must present their own contributions to their project, so that they can be evaluated individually. The presentation should be aimed at NLP colleagues, so highlight which approaches and techniques you used, which datasets you used, and the insights obtained. Presentation slides must be pushed to your project repository the day before the presentations. The schedule of presentations will be announced via TUWEL, please attend all presentations in your section.

## Timetable
- **03.10.2025** — Project topics announced  
- **10.10.2025** — Milestone 1 introduced  
- **12.10.2025, 23:55** — Group registration & topic selection deadline  
- **17.10.2025** — Milestone 2 introduced  
- **02.11.2025, 23:55** — Milestone 1: Dataset preprocessing due  
- **30.11.2025, 23:55** — Milestone 2: Baseline implementations due  
- **15.12.2025, 13–17h** — Review Meeting I 
- **19.12.2025, 9–13h** — Review Meeting II (Token13 slot) (Instructor: Gabor Recski, review meetings: 12/19/2025, 9-13h)  
- **15.01.2026, 23:55** — Presentation slides due (upload to GitHub)  
- **16.01.2026** — Final presentations  
- **25.01.2026, 23:55** — Final submission deadline (code + report)



