# Machine learning project planning checklist

## Questions about the task

<div class="row">Describe the task at a high level.<div class="dots"></div></div>

<div class="row"><div class="dots"></div></div>

<div class="row">What target(s) will you predict?<div class="dots"></div> &nbsp; ▢ Multi-target task</div>

<div class="row">How many feature(s) will you use?<div class="dots"></div></div>

What kind of task is being performed? **Choose one from each row.**
<div class="row">&nbsp; &nbsp; **Supervision:** &nbsp; &nbsp; ▢ Supervised &nbsp; &nbsp; ▢ Semi-supervised &nbsp; &nbsp; ▢ Unsupervised &nbsp; &nbsp; ▢ Other<div class="dots"></div></div>

<div class="row">&nbsp; &nbsp; **Supervised tasks:** &nbsp; &nbsp; ▢ Classification &nbsp; &nbsp; ▢ Regression &nbsp; &nbsp; ▢ Segmentation &nbsp; &nbsp; ▢ Other<div class="dots"></div></div>

<div class="row">&nbsp; &nbsp; **Classification tasks:** &nbsp; &nbsp; ▢ Binary &nbsp; &nbsp; ▢ Multi-class &nbsp; &nbsp; ▢ Other<div class="dots"></div></div>

<div class="row">What is the current company or industry state-of-the-art on this task?<div class="dots"></div></div>

<div class="row"><div class="dots"></div></div>

<div class="row">What level of performance would be satisfactory on this task?<div class="dots"></div></div>


## Questions about the dataset

<div class="row">Where will the data come from? List all potential sources.<div class="dots"></div></div>

<div class="row"><div class="dots"></div></div>

<div class="row">How much data cleaning will likely be required?<div class="dots"></div></div>

<div class="row"><div class="dots"></div></div>

<div class="row">What will you do about missing data? &nbsp; &nbsp; &nbsp; &nbsp; ▢ Nothing &nbsp; &nbsp; &nbsp; &nbsp; ▢ Dropped &nbsp; &nbsp; &nbsp; &nbsp; ▢ Mean &nbsp; &nbsp; &nbsp; &nbsp; ▢ Regression</div>


## Questions about the features

<div class="row">What are the features in this task?<div class="dots"></div></div>

<div class="row"><div class="dots"></div></div>

<div class="row">Will you engineer any new features? If so, how?  &nbsp; &nbsp; **YES &nbsp; &nbsp; NO**</div>

<div class="row"><div class="dots"></div></div>

<div class="row">What other features could be considered in the future?<div class="dots"></div></div>

<div class="row"><div class="dots"></div></div>


## Questions about the label(s) or target(s)

<div class="row">What is (or are) the label(s) or target(s) in this task?<div class="dots"></div></div>

<div class="row">Where will the target or labels come from, and how reliable are they?<div class="dots"></div></div>

<div class="row"><div class="dots"></div></div>


## Questions about training and evaluation

<div class="row">What kinds of models seem most likely to work?<div class="dots"></div></div>

<div class="row">What kinds of cost functions and metrics will make sense?<div class="dots"></div></div>

<footer>v0.3  •   © 2023 Matt Hall  •  Licensed CC BY 4.0</footer>

---

# Annotations


## Questions about the task


### T.1 Describe the task at a high level.

Give the 'headline' description of the project. Include any aspirational 'stretch' goals. Be as succinct as possible, but capture the business value of the task if possible.


#### Example

"Integrated porosity: beyond PHIND. Our goal is to predict porosity using as many diver inputs as possible."


### T.2 What target(s) will you predict?

The target or 'labels' are what the model predicts, often symbolically denoted by y. Usually, there is a single target value (for regression problems) or label (for classification problems) for each record ('instance', 'example', or row) in the data. If there is more than one target or label, then the task is 'multi-target' or 'multi-output' (see [the scikit-learn documentation](https://scikit-learn.org/stable/modules/multiclass.html) for more information).


#### Example

If a model is predicting porosity from wireline logs, seismic impedance, and core descriptions, then porosity is the target.


### T.3 How many feature(s) will you use?

The features are the predictors, or inputs to the model, often symbolically denoted by X. The number of features corresponds to the dimensionality _M_ of the task; that is, each record is a vector of _M_ dimensions. Each feature is usually represented as a column in the input dataset.


#### Example

If a model is predicting porosity from GR, RHOB, and SP logs, then there are three features.


### T.4 What kind of task is being performed?

Fist, you need to decide if the task is supervised or unsupervised. Most predictive tasks are supervised: you provide the target or label for each data record to the training process. Clustering tasks are unsupervised — the records are unlabeled and the goal is to find structure in the data. In semi-supservised tasks, a (usually small) subset of the points are labeled; unsupervised methods are combined with supervised ones to build better models than would be possible with only the labeled data. There are other types of machine learning besides these, for example online learning or reinforcement learning.

Most supervised machine learning tasks can be further divided into a three major types:

- **Classification** — the target is a set of two or more discrete categories, like {_pay, non-pay_} ('binary' classification), or {_sandstone, limestone, shale_} ('multiclass').
- **Regression** — the target is a continuous property, such as _porosity_ or _net present value_.
- **Segmentation** — the task is to partition an image into 'image objects'. For example, predicting faults from seismic data, or identifying rock types in core photos.


#### Example

If a model is predicting porosity from GR, RHOB, and SP logs, then this is a supervised regression task. On the other hand, a model predicting high, medium or low bit wear from partially labeled drilling data is a  semi-supervised multiclass classification task.


### T.5 What is the current company or industry state-of-the-art on this task?

Think about how this task is typically solved in the industry today (assuming it's not using machine learning already). Is it completely manual process? Do people use numerical modeling? What kinds of information ('features') does a geoscientist need to solve this task, for example? What is the output of the task? For example, does it produce a map, or a map plus uncertainty?


#### Example

Porosity is often predicted by applying numerical models to wireline logs, but using knowledge of the well and the basin in a qualitative way. How can your model get access to that kind of information too?


### T.6 What level of performance would be satisfactory on this task?_

Consider what the usual, 'state of the art' performance level is on the task. Also think about what would be an acceptable level of performance. Again, it is much easier to do this before you begin.


Knowing what kind of performance you are looking for will help you know when you have met your requirements.


#### Example

A petrophysicist can construct a lithology model that is 85% accurate for a given zone, but it takes one day per well. You decide that 80% is adequate for your needs, as long as you can process 100 wells in an hour.


## Questions about the dataset


### D.1 Where will the data come from?

You should document the provenance of the data you will attempt to use to train the model, along with any notes about the version, date, author(s) and owner of the dataset. This is important for reproducibility, and for keeping track of future versions of your model.


#### Example

You will get the data from an in-house well database. You add detailed comments to the script that extracted the data, and keep it under version control.


### D.2 How much data cleaning will likely be required?

Data cleaning and reprocessing steps might include:

- Dropping features, or combining features from multiple datasets.
- Dealing with malformed or non-standard data elements.
- Converting between units of physical properties.
- Regularizing dates, times, currencies, etc, so they can be used as data.
- Encoding features or targets, e.g. binning, integer encoding, or one-hot encoding.

Two more steps that some consider to be part of the preprocessing pipeline are handling missing data and scaling the data. These are addressed in the next two questions.


#### Example

You are predicting a porosity from logs. You combine datasets containing logs with SI and Imperial units, so you convert everything to SI units. You have both slowness and velocity logs, so you drop velocity in favour of slowness. You convert (UTMx, UTMy) locations to the same CRS, then add them as features to each row.


### D.3 What will you do about missing data?_

In general, predictive models do not cope with missing data. So either you or the tool will probably need to deal with it. Check the documentation to find out if and how the tool handles missing data internally.


Common strategies for handling missing data include:

- Dropping records with one or more missing features. This is easy.
- Replacing missing values with the mean of that feature.
- In a timeseries, replacing missing values with the previous one(s).
- Predicting missing values from other values, typically with a regression model.


Some models combine these techniques — see the example below.


#### Example

You are predicting the DTS log from other logs. Some wells are missing a DTP log, so you make a linear model to predict DTP in those wells ('regression imputation'). Next, some logs have missing intervals; since you have a lot of data, you drop all rows with missing values.


## Questions about the features


### F.1 What are the features in this task?

The features are the predictors, or inputs to the model, often symbolically denoted by X. List them here (or, if there are too many, generalize about them).


#### Example

If a model is predicting porosity from GR, RHOB, and SP logs, then those logs are the features. If you are classifying core images, then the core images (specifically, their pixels) are the features, along with any other attribute you include, such as depth or wireline log values. 


### F.2 Will you engineer any new features? If so, how?

It is often advantageous to compute new, more predictive features from the initial ones. In general, this is called 'feature engineering'. People use various techniques, for example:

- Non-linear transformations of features.
- Combining features in so-called interactions (for example, the product of two features).
- Computing 'flags' in space or time, such as off-gauge hole from the caliper log, or weekends from dates.
- Smoothing features in space or time.


#### Example

You are predicting lithology from well logs. You replace the resistivity logs with their logarithms, because this makes their distributions easier to visualize and, in some models, use. 


### F.3 What other features could be considered in the future?

What other features could help improve the prediction? These might be difficult to integrate at this point, but it's a good idea to think about what they might be. Sometimes there is easily accessible data that could help too — for example timestamps or location data on images, or cuttings data for lithology prediction. Try asking yourself what a human might look for if they were given the task.


#### Example

You are predicting lithology from logs. You have extracted lithology labels from the cuttings descriptions in the mud log, and you have triplec-combo wireline data. But some of the wells have image logs, and still others have core, but these data are not easily accessible. You note that they would probably help to constrain the task, and decide to investigate these ideas in the event the simpler proof-of-concept is successful.


## Questions about the label(s) or target(s)


### L.1 What is (or are) the target(s) or label(s) in this task?

List the target(s) or label(s). See T.1.


#### Example

If you are predicting lithology from well logs, the labels are the lithologies ('sandstone', 'shale', etc). If you are predicting a DTS log from other well logs, the target is the DTS log. If you are categorizing photographs of drilling bits, the labels might be 1 for 'worn' and 0 for 'not worn'.


### L.2 Where will the target or labels come from, and how reliable are they?

This question is asking about the ground truth data, for example did you assign the labels yourself, or were they already present in the dataset? If they are interpretations, who did them? Have you checked the labels on some of the training and validation examples? Is it possible that some of them are incorrectly or incompletely labeled?


If this is a regression problem, where is the target dataset coming from? 


Often the labels we have are the result of an interpretation and therefore prone to error, bias, and uncertainty. For example, we may have fault interpretations in seismic data — these are very difficult to corroborate objectively, but we could collect more opinions to try to understand the uncertainty.


#### Example

You are predicting lithology from well logs. The lithologies were interpreted by 3 different sedimentologists. You check a random sample of the wells and find that some wells have fine and medium siltstone, while others only have medium siltstone. If you think that variance might just be the result of different geologists logging the core, those labels should be combined into 'siltstone'. 


## Questions about training and evaluation


### E.1 What kinds of models seem most likely to work?

Sometimes we suspect that certain models will work better than others, perhaps because we've worked on the problem before, or because others have had success with certain models.


#### Example

Faced with predicting a well log from other well logs you might choose to start with linear regression because it is a standard approach to this task. Or to attempt an image classification task, you might try a support vector machine as a simple approach, then a deep convolutional neural network because they are well-suited to the task.


### E.2 What kinds of cost functions and metrics will make sense?

It's a good idea to think about how you will evaluate a model's performance, so you can ensure that the metric(s) you choose match you and your colleagues' expectations of the model. You should also ensure that the cost function(s) used in training reflect your opinion of error and accuracy. 


#### Example

You are predicting DTS from other well logs using linear regression. Internally this uses squared error, which is reasonable. You use validation evaluation to compare between models. You also judge that the result looks reasonable when plotted with measured DTS logs from the area. Finally you use RMS error — which is in the same units as the log and therefore more intuitive than mean squared error — to determine accuracy and compare between models.

---

# Additional information


## Changelog


### v0.3, July 2023

- Moved the file to [https://github.com/scienxlab/checklists](https://github.com/scienxlab/checklists) in order to be able to keep multiple checklists together and eventually create a web page for them.
- Changed the format to [Markdown](https://www.markdownguide.org/basic-syntax/) plus HTML, with styles applied via CSS using a tool like Pandoc. Instructions for conversion added to the README, but eventually we'll publish them on the aforementioned web page.


### v0.2, October 2021

- Forked from original document and slimmed down to deal only with planning projects.


### v0.1, June 2021

- Original document, started as _Machine learning tool evaluation checklist_
