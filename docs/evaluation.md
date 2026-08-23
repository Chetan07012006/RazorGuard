# Evaluation Report

## Dataset profile

These values were measured from the downloaded IEEE-CIS training files:

| Quantity | Value |
|---|---:|
| Transactions | 590,540 |
| Identity records | 144,233 |
| Fraud records | 20,663 |
| Fraud rate | 3.4990% |

## Validation design

The labeled transactions are sorted by `TransactionDT`. The earliest 80% (472,432 rows) are used for training and the latest 20% (118,108 rows) are held out. The validation fraud rate is 3.4409%.

## Model comparison

| Model | Validation PR-AUC | Fraud F1 | Threshold |
|---|---:|---:|---:|
| Class-balanced logistic baseline | 0.189986 | 0.292406 | 0.80 |
| Identity-enriched logistic model | 0.187334 | 0.288786 | 0.80 |
| Class-weighted histogram gradient boosting | 0.457308 | 0.428455 | 0.80 |

For the tree model at threshold `0.80`, fraud precision is 0.407227 and fraud recall is 0.452018. These are held-out validation results, not Kaggle leaderboard results.
