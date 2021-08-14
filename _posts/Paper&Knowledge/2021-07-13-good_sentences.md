---
title:  "논문들 좋은 문장 모음"
excerpt: "논문들 좋은 문장 모음"
categories:
  - Paper & Knowledge
  
tags:
  - Paper & Knowledge
 
published: false
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-08-11T11:19:00-05:00
---

# Abstract

1. Time series outlier detection has been extensively studied with many advanced algorithms propsed in the past decade. [Revisiting time series outlier detection: definitions and benchmarks.]

2. <u>Learning similarity functions between image pairs with deep neural networks yields highly correlated activations of large embeddings.</u> [BIER-Boosting Independent Embeddings Robustly.]

3. <u>Our method does not introduce any additional parameters and works with any differentiable loss function.</u> [BIER-Boosting Independent Embeddings Robustly.]

4. We evaluate our metric learning method on image retrieval tasks and show that it improves over state-of-the-art methods on Cars-196 and VehicleID datasets <u>by 
a significant margin</u> [BIER-Boosting Independent Embeddings Robustly.]

5. <u>Riding on the waves of deep neural networks,</u> deep metric learning has achieved promising results in various tasks by using triplet network or Siamese network. [Hard-Aware Deply Cascaded Embedding.]

# Introduction
    
1. CCTV surveilance and the facial recognition technology is <u>on its way</u> to becoming ubiquitous in large cities around the world. [Understanding the Privacy
of Facial Recognition Embeddings.]

2. In this paper, <u>our goal is twofold</u>; we investigate ...

3. <u>With the hope that</u> these insights could motivate future work, we have open-sourced all the datasets, the pre-processing and synthetic scripts, and the algorithm implementation in TODS. 
[Revisiting time series outlier detection: definitions and benchmarks.]

4. However, <u>their unstructured nature</u> combined with the complexity and ambiguity of natural language <u>pose</u> a challenge when using radiology reports for clinical research and other downstream applications, 
especially in settings with limited labeled data. [RadGraph: Extracting Clinical Entities and Relations from Radiology Reports.]

5. We define a novel information extraction schema for radiology reports, <u>intended to cover modst climically relevant information with in the report while allowing for ease and consistency during annotation.</u> 
[RadGraph: Extracting Clinical Entities and Relations from Radiology Reports.]

6. <u>Along with these datasets</u>, there have been many advancements in NLP for the task of entiy and relation extraction. [RadGraph: Extracting Clinical Entities and Relations from Radiology Reports.]

7. <u>To address this issue</u>, we present a learning approach, called BIER. [BIER-Boosting Independent Embeddings Robustly.]

8. <u>In our evaluation,</u> we show that BIER significantly reduces the correlation of large embeddings and works with several loss function <u>while increasing retrieval accuracy
by a large margin.</u> [BIER-Boosting Independent Embeddings Robustly.]

9. <u>BIER does not introduce any additional parameters into a CNN and has only negligible additional cost during training time and runtime.</u> [BIER-Boosting Independent Embeddings Robustly.]

10. Moreover, the commonly available class assignments <u>give rise to</u> image relations aside from the standard, supervised learning task of "pulling" smaples with identical class labels together
while "pushing" away samples with different labels. [DIVA: Diverse Visual Feature Aggregation for Deep Metric Learning]

11. We <u>tackle the issue</u> of generalization in DML by designing diverse learning tasks complementing standard supervised training, leveraging only the comonly
provided training samples and labels. [DIVA: Diverse Visual Feature Aggregation for Deep Metric Learning]

# Background

1. Figure.3 illustrates the thress types of outliers that often serve as a <u>de-facto-standard</u>: [Revisiting time series outlier detection: definitions and benchmarks.]


# Related work

1. A <u>central limitation</u> of both of these approaches is that they require task-specific datasets to be densely annotated by domain experts. [RadGraph: Extracting Clinical Entities and Relations from Radiology Reports.]

2. Image embedding falls <u>under the unbrella</u> of distance metric learning. [Deep Randomized Ensembles for Metric Learning.]

3. THe main objective of metric learning in Computer Vision is to learn a function which maps a k-dimensional input vector, which is typically an input image or a feature representation of an image, 
into a d-dimensional vector space. [BIER-Boosting Independent Embeddings Robustly.] -> related work 맨 처음에 사용할때 좋다. 

4. They <u>leverage</u> the benefits of deeply supervised networks <u>by employing</u> a contrastive loss function and train lower layers of the network to handle easier examples, 
and higher layers in a network to handle harder examples. [BIER-Boosting Independent Embeddings Robustly.]

# Methods

1. <u>In what follows</u>, we first describe the details of the synthetic datasets and the real-world datasets, and then <u>elaborate</u> on the included algorithms. [Revisiting time series outlier detection: definitions and benchmarks.]

2. <u>As opposed to learning a distance metric,</u> in our work we learn a cosine similarity score which we define as dot product between two embeddings. [BIER-Boosting Independent Embeddings Robustly.]






# Experimental Results

1. Due to the space limitation, the detailed benchmark results of synthetic datasets <u>are tabulated</u> in Appendix D. [Revisiting time series outlier detection: definitions and benchmarks.]

2. Except for the web attack dataset, all of other datasets <u>are dominated by</u> AR, IForest and OCSVM.

3. We <u>report</u> the full results for the benchmarks across entity types in Table 4 and across relation types in Table 5.








# Discussion

1. <u>Given that existing information extraction systems for radiology reports often suffer from a lack of report coverage,</uz> we measure the number of tokens and sentances in report sections covered by our schema.
 [RadGraph: Extracting Clinical Entities and Relations from Radiology Reports.]












# Conclusion






