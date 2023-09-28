| Category     | Pitfall                                      | Severity | Consequences                               | Mitigation                                 |
|--------------|----------------------------------------------|:--------:|--------------------------------------------|--------------------------------------------|
| Design       | Asking the wrong question                    |     ▲    | Useless model, low trust                   | Consult user/customer, ask more questions  |
| Design       | Identifying the wrong task                   |     ▲    | Poor performance, low trust                | Challenge assumptions                      |
| Design       | No baseline/SOTA performance                 |     ●    | Poor performance, low trust                | Measure SOTA performance                   |
| Design       | No success criterion                         |     ●    | Poor performance, low trust                | Ask user/customer what success means       |
| Data         | Poor quality features                        |     ●    | Poor performance                           | Find better data                           |
| Data         | Poor quality labels                          |     ●    | Poor performance                           | Find or make better labels                 |
| Data         | Unrecognized non-independent records         |     ●    | Leads to leakage via improper splitting    | Careful splitting, exploit correlation     |
| Data         | Unrecognized class imbalance                 |     ●    | Poor performance on minority classes       | Balance classes, better evaluation metrics |
| Data         | Hidden stratification                        |     ●    | Poor performance on important examples     | Ask user/customer, monitor performance     |
| Data         | Data not representative                      |     ●    | Leads to out-of-distribution application   | Find representative data                   |
| Data         | Spurious/noncausal correlations              |     ●    | Leads to leakage                           | Pay attention to causality                 |
| Data         | Missing explanatory variables                |     ●    | Poor performance                           | Pay attention to causality                 |
| Leakage      | Using features not available in application  |     ▲    | Overoptimism                               | Examine real application data              |
| Leakage      | Improper splitting: scaling                  |     ●    | Overoptimism                               | Code review                                |
| Leakage      | Improper splitting: correlations             |     ●    | Overoptimism                               | Code review                                |
| Leakage      | Improper splitting: augmentation             |     ●    | Overoptimism                               | Code review                                |
| Modeling     | Poor choice of algorithm                     |     ●    | Poor performance, poor explainability      | Develop understanding, code review         |
| Modeling     | Lack of understanding of algorithm           |     ●    | Poor performance, poor explainability      | Develop understanding                      |
| Modeling     | No or inappropriate feature scaling          |     ●    | Poor performance, no convergence           | Develop understanding, code review         |
| Modeling     | Inappropriate feature encoding               |     ●    | Poor performance                           | Develop understanding, code review         |
| Modeling     | Poor preprocessing: imputation               |     ●    | Poor performance                           | Develop understanding, code review         |
| Modeling     | Poor preprocessing: outlier elimination      |     ●    | Overoptimism, poor explainability          | Develop understanding, code review         |
| Modeling     | Poor hyperparameter tuning                   |     ●    | Poor performance, slow or no convergence   | Develop understanding, code review         |
| Underfitting | No or inappropriate basis expansion          |     ●    | Poor performance                           | Develop understanding, code review         |
| Underfitting | No data augmentation                         |     ●    | Poor performance                           | Develop understanding, code review         |
| Overfitting  | Too many parameters                          |     ●    | Overoptimism, poor explainability          | Choose simpler algorithms                  |
| Overfitting  | Too many features                            |     ▼    | Overoptimism, poor explainability          | Dimensionality reduction                   |
| Overfitting  | No regularization                            |     ●    | Overoptimism, poor performance             | Use regularization, or equivalent          |
| Overfitting  | Insufficient data                            |     ▼    | Overoptimism, poor performance             | Get more data                              |
| Evaluation   | Over-using test set                          |     ●    | Leads to overfitting                       | Use more splits                            |
| Evaluation   | Wrong metric                                 |     ●    | Overoptimism, overlooking minority classes | Choose better metric                       |
| Evaluation   | Not looking at variance                      |     ●    | Poor understanding of performance          | Use folded cross-validation                |
| Evaluation   | Not evaluating residuals (in regression)     |     ●    | Spurious model                             | Examine residuals                          |
| Evaluation   | Not comparing train and test scores          |     ●    | Poor understanding of performance          | Compute training scores                    |
| Application  | Unscaled input                               |     ▲    | Wildly spurious predictions                | Integrate scaler into modeling pipeline    |
| Application  | Covariate (feature) shift: new P(X)          |     ▲    | Poor performance                           | Monitor incoming feature distribution      |
| Application  | Label shift: new P(y)                        |     ▲    | Poor performance                           | Test future model performance              |
| Application  | Concept drift (posterior shift): new P(y\|X) |     ▲    | Poor performance                           | Monitor model performance                  |
| Application  | Nonstationary input (in time or space)       |     ●    | Poor performance, drift over time or space | Test future or local model performance     |
| Deployment   | No consideration of complex system           |     ▲    | Unintended consequences, low trust         | Talk to users/customers                    |
| Deployment   | No contingency                               |     ●    | Low trust                                  | Plan ahead with users/customers            |
| Deployment   | No training of users                         |     ▲    | Low impact, low trust                      | Train users/customers                      |
| Deployment   | No documentation                             |     ●    | Inappropriate application, low trust       | Document the modeling process & product    |
| Engineering  | No code or version control                   |     ▲    | Technical debt, high chance of error       | Use version control                        |
| Engineering  | No data versioning                           |     ▲    | High chance of error                       | Use version control                        |
| Engineering  | No tests of critical code                    |     ▲    | High chance of error                       | Write tests                                |
| Governance   | No quality control                           |     ▲    | High chance of error                       | Integrate QC into development process      |
| Governance   | Using protected features                     |     ▲    | Potential unfair bias                      | Removed protected features                 |
| Governance   | Violates regulations or ethics               |     ▲    | Legal risk                                 | Consult professional services              |
