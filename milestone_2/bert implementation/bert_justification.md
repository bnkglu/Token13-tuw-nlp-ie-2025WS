Milestone 2 Report: Machine Learning Baseline
Topic 8: Explainable Relation Extraction

1. Introduction
The goal of Topic 8 is to develop explainable rule-based systems for Relation Extraction (RE), specifically by "leveraging machine learning models for creating patterns".

For Milestone 2, our objective was to implement and evaluate a Machine Learning baseline. We selected a Deep Learning approach using RoBERTa-base to establish a high-performance "upper bound" for the task. This baseline serves two purposes: it provides a quantitative benchmark for our future rule-based system, and it acts as a "teacher" model to generate high-confidence examples for pattern mining in Milestone 3.

2. Methodology: The RoBERTa Model
2.1 Architecture Selection

We utilized RoBERTa (Robustly optimized BERT approach), a Transformer-based language model. Unlike traditional statistical models (like Naive Bayes or Logistic Regression) that treat sentences as a "bag of words," RoBERTa utilizes an Attention Mechanism. This allows the model to understand the specific context of words based on their neighbors, which is crucial for distinguishing complex relations.

2.2 Input Representation (Entity Markers)

To adapt the language model for Relation Extraction, we employed an Entity Marker strategy. We modified the raw text to explicitly highlight the subject and object entities using special tokens:

Original: "The audit was about waste."

Processed: "The [E1]audit[/E1] was about [E2]waste[/E2]."

This technique guides the model's self-attention mechanism to focus specifically on the grammatical path and context connecting the two marked entities.

2.3 Training Configuration

The model was fine-tuned using the simpletransformers library with the following hyperparameters:

Base Model: roberta-base (12 layers, 768 hidden units).

Epochs: 4 (Iterating over the full dataset 4 times).

Batch Size: 16.

Learning Rate: 3e 
−5
 .

3. Quantitative Evaluation
The baseline was evaluated on the held-out test set (N=2717). The model achieved strong performance, confirming that the dataset contains learnable linguistic patterns.

Overall Metrics:

Accuracy: 85%

Macro F1-Score: 0.86

Detailed Classification Report:

Relation Class	Precision	Recall	F1-Score	Analysis
Cause-Effect	0.94	0.95	0.94	Best Performer. The model effectively identifies causal language (e.g., "caused by", "due to"), suggesting these relations have very distinct syntactic patterns.
Entity-Destination	0.91	0.95	0.93	High performance indicates that movement/destination prepositions are easily learned by the Transformer.
Message-Topic	0.85	0.95	0.90	The model shows high recall, successfully retrieving most topic-based relations.
Other	0.69	0.57	0.62	Lowest Performer. This is expected, as "Other" is a negative class containing diverse examples that do not fit the predefined schemas, making it harder to generalize.
4. Integration with Milestone 3 (Future Work)
As per the assignment requirements to "leverage machine learning models for creating patterns", this baseline is not a standalone solution but a component of our final pipeline.

Strategy: High-Confidence Filtering We utilized the trained RoBERTa model to filter the test data. We extracted 1,970 instances where the model's prediction confidence exceeded 95% and was correct.

Logic: If the model is >95% confident, the sentence likely contains a strong, unambiguous linguistic signal (a clear pattern).

Next Step: These filtered examples will be fed into the POTATO library (Rule-based system) in Milestone 3 to automatically generate explainable graph patterns. This allows us to combine the high precision of rules with the pattern-recognition capabilities of Deep Learning.

5. Conclusion
We have successfully implemented a state-of-the-art ML baseline that exceeds the performance requirements for Milestone 2. The high F1-score (0.86) validates our data processing strategy and provides a robust foundation for the explainability work in the final phase of the project.