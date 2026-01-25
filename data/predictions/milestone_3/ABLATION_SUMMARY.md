# Ablation Study Results

## Experiment Configurations

| # | Config Name | FrameNet & WordNet | Prediction Mode | Description |
|---|-------------|--------------------|-----------------| ------------|
| 1 | Baseline (FN & WN OFF) | OFF | first_match | Statistical system (Semantics OFF) |
| 2 | FN & WN OFF + priority | OFF | priority_based | Statistical system (Priority Ranking) |
| 3 | FN & WN ON + first_match | ON | first_match | Statistical system (Semantics ON) |
| 4 | FN & WN ON + priority | ON | priority_based | Statistical system (Priority & Semantics) |

### Prediction Modes:
- **first_match** : Rules ranked by precision & support, first matching rule wins
- **priority_based** : Rules ranked by pattern type (PREP_STRUCT > LEXNAME > BIGRAM > SYNSET/FRAME/LEMMA)

## Summary Statistics (Test Set)

| Configuration | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---------------|----------|-----------------|--------------|----------|
| 01_fn_wn_off_first_match | 0.500 | 0.593 | 0.402 | 0.439 |
| 02_fn_wn_off_priority_based | 0.509 | 0.624 | 0.411 | 0.449 |
| 03_fn_wn_on_first_match | 0.590 | 0.638 | 0.510 | 0.549 |
| 04_fn_wn_on_priority_based | 0.601 | 0.661 | 0.520 | 0.560 |

## Detailed Results

### 01_fn_wn_off_first_match

```
### TEST SET RESULTS ###
                           precision    recall  f1-score   support

      Cause-Effect(e1,e2)      0.843     0.724     0.779       134
      Cause-Effect(e2,e1)      0.753     0.737     0.745       194
   Component-Whole(e1,e2)      0.389     0.043     0.078       162
   Component-Whole(e2,e1)      0.707     0.353     0.471       150
 Content-Container(e1,e2)      0.635     0.830     0.720       153
 Content-Container(e2,e1)      0.867     0.333     0.481        39
Entity-Destination(e1,e2)      0.724     0.866     0.789       291
Entity-Destination(e2,e1)      0.000     0.000     0.000         1
     Entity-Origin(e1,e2)      0.800     0.682     0.737       211
     Entity-Origin(e2,e1)      0.600     0.064     0.115        47
 Instrument-Agency(e1,e2)      0.467     0.318     0.378        22
 Instrument-Agency(e2,e1)      0.593     0.478     0.529       134
 Member-Collection(e1,e2)      0.444     0.125     0.195        32
 Member-Collection(e2,e1)      0.575     0.114     0.191       201
     Message-Topic(e1,e2)      0.696     0.567     0.625       210
     Message-Topic(e2,e1)      0.724     0.412     0.525        51
                    Other      0.201     0.489     0.285       454
  Product-Producer(e1,e2)      0.593     0.148     0.237       108
  Product-Producer(e2,e1)      0.657     0.358     0.463       123

                 accuracy                          0.500      2717
                macro avg      0.593     0.402     0.439      2717
             weighted avg      0.591     0.500     0.495      2717
```

### 02_fn_wn_off_priority_based

```
### TEST SET RESULTS ###
                           precision    recall  f1-score   support

      Cause-Effect(e1,e2)      0.843     0.724     0.779       134
      Cause-Effect(e2,e1)      0.747     0.747     0.747       194
   Component-Whole(e1,e2)      0.375     0.037     0.067       162
   Component-Whole(e2,e1)      0.701     0.360     0.476       150
 Content-Container(e1,e2)      0.654     0.889     0.753       153
 Content-Container(e2,e1)      0.875     0.359     0.509        39
Entity-Destination(e1,e2)      0.742     0.880     0.805       291
Entity-Destination(e2,e1)      0.000     0.000     0.000         1
     Entity-Origin(e1,e2)      0.800     0.701     0.747       211
     Entity-Origin(e2,e1)      0.600     0.064     0.115        47
 Instrument-Agency(e1,e2)      0.467     0.318     0.378        22
 Instrument-Agency(e2,e1)      0.620     0.500     0.554       134
 Member-Collection(e1,e2)      0.800     0.125     0.216        32
 Member-Collection(e2,e1)      0.575     0.114     0.191       201
     Message-Topic(e1,e2)      0.707     0.562     0.626       210
     Message-Topic(e2,e1)      0.786     0.431     0.557        51
                    Other      0.202     0.491     0.286       454
  Product-Producer(e1,e2)      0.654     0.157     0.254       108
  Product-Producer(e2,e1)      0.698     0.358     0.473       123

                 accuracy                          0.509      2717
                macro avg      0.624     0.411     0.449      2717
             weighted avg      0.604     0.509     0.503      2717
```

### 03_fn_wn_on_first_match

```
### TEST SET RESULTS ###
                           precision    recall  f1-score   support

      Cause-Effect(e1,e2)      0.860     0.731     0.790       134
      Cause-Effect(e2,e1)      0.715     0.840     0.773       194
   Component-Whole(e1,e2)      0.726     0.377     0.496       162
   Component-Whole(e2,e1)      0.674     0.387     0.492       150
 Content-Container(e1,e2)      0.658     0.830     0.734       153
 Content-Container(e2,e1)      0.714     0.513     0.597        39
Entity-Destination(e1,e2)      0.746     0.907     0.819       291
Entity-Destination(e2,e1)      0.000     0.000     0.000         1
     Entity-Origin(e1,e2)      0.778     0.796     0.787       211
     Entity-Origin(e2,e1)      0.800     0.255     0.387        47
 Instrument-Agency(e1,e2)      0.583     0.318     0.412        22
 Instrument-Agency(e2,e1)      0.575     0.485     0.526       134
 Member-Collection(e1,e2)      0.667     0.375     0.480        32
 Member-Collection(e2,e1)      0.708     0.592     0.645       201
     Message-Topic(e1,e2)      0.686     0.633     0.658       210
     Message-Topic(e2,e1)      0.634     0.510     0.565        51
                    Other      0.252     0.403     0.310       454
  Product-Producer(e1,e2)      0.704     0.352     0.469       108
  Product-Producer(e2,e1)      0.649     0.390     0.487       123

                 accuracy                          0.590      2717
                macro avg      0.638     0.510     0.549      2717
             weighted avg      0.633     0.590     0.593      2717
```

### 04_fn_wn_on_priority_based

```
### TEST SET RESULTS ###
                           precision    recall  f1-score   support

      Cause-Effect(e1,e2)      0.873     0.716     0.787       134
      Cause-Effect(e2,e1)      0.700     0.876     0.778       194
   Component-Whole(e1,e2)      0.753     0.377     0.502       162
   Component-Whole(e2,e1)      0.667     0.400     0.500       150
 Content-Container(e1,e2)      0.672     0.895     0.768       153
 Content-Container(e2,e1)      0.724     0.538     0.618        39
Entity-Destination(e1,e2)      0.764     0.914     0.833       291
Entity-Destination(e2,e1)      0.000     0.000     0.000         1
     Entity-Origin(e1,e2)      0.797     0.801     0.799       211
     Entity-Origin(e2,e1)      0.750     0.255     0.381        47
 Instrument-Agency(e1,e2)      0.538     0.318     0.400        22
 Instrument-Agency(e2,e1)      0.621     0.537     0.576       134
 Member-Collection(e1,e2)      0.857     0.375     0.522        32
 Member-Collection(e2,e1)      0.721     0.592     0.650       201
     Message-Topic(e1,e2)      0.707     0.633     0.668       210
     Message-Topic(e2,e1)      0.730     0.529     0.614        51
                    Other      0.259     0.421     0.321       454
  Product-Producer(e1,e2)      0.723     0.315     0.439       108
  Product-Producer(e2,e1)      0.701     0.382     0.495       123

                 accuracy                          0.601      2717
                macro avg      0.661     0.520     0.560      2717
             weighted avg      0.650     0.601     0.604      2717
```

---
Generated: Sat Jan 24 16:20:47 CET 2026

## Comparative Analysis
Analysis of agreement between **first_match** (Baseline) and **priority_based** strategies.

| Comparison | Same Rule % | Different Rule % | Label Agreement (Overall) | Label Agreement (Diff Rule) |
|------------|-------------|------------------|---------------------------|-----------------------------|
| FN/WN OFF (Baseline vs Priority) | 87.2% | 12.8% | 97.2% | 78.4% |
| FN/WN ON (First vs Priority) | 71.8% | 28.2% | 95.1% | 82.7% |

### Key Insights
- **Different Rule %**: Shows how often `priority_based` picked a different rule than `first_match`.
- **Label Agreement (Diff Rule)**: Shows how often they agreed on the label *even when choosing different rules*.
- High agreement (99%+) implies the system is robust: Syntactic and Semantic rules reinforce each other.

