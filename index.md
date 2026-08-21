# by Sara Oliveira

---

## Research Scholarship Project: Anomaly Detection in Waste Transport Networks
*EnSafe Project — INESC TEC and IGAMAOT*

Developed an anomaly detection system to identify fraudulent activity in the Portuguese waste management network. The project aimed to give inspectors at IGAMAOT (Portugal's General Inspectorate of Agriculture, Sea, Environment and Spatial Planning) an efficient analytical tool to prioritize inspections, speed up the detection of illicit behavior, and support regulatory compliance. Using real waste transport data, I engineered features capturing companies' relational, operational, and temporal behavior, then developed and evaluated a hybrid detection framework combining statistical change-detection methods (Page-Hinkley, CUSUM) with deep learning models (LSTM-VAE, GCN) to detect anomalies such as abrupt activity shifts and unusual connectivity patterns (such as collusive triangles). The methods were validated against a small set of companies confirmed as fraudulent by IGAMAOT inspectors, showing that detecting abrupt or intermittent behavioral changes together with complex connectivity structures (such as triangles) meaningfully improves fraud detection in the waste management network.

**Techniques:** network analysis, graph neural networks (GCN), LSTM-VAE, statistical change detection (Page-Hinkley, CUSUM), anomaly/fraud detection, feature engineering

**Anomaly detection interactive platform:** the project resulted in a user-friendly platform integrating the statistical methods and AI models I developed with graph analytics and interactive visualization, enabling IGAMAOT inspectors to use the methodology directly, to identify suspicious companies that should be prioritized in the annual inspection plan, and to analyse the activity of companies they have themselves deemed as suspicious.

[Presentation](https://github.com/saratoninb9/Data-Science-Portfolio/blob/3ac7f73cfcdc270d9b7c1032e5454ff52999d924/AnomalyDetectionplatform_Presentation.pdf) · [Data Pipeline Code](https://github.com/saratoninb9/Data-Science-Portfolio/blob/3ac7f73cfcdc270d9b7c1032e5454ff52999d924/Data_pipeline.py)

### Publications

This project resulted in 2 publications in international peer-reviewed conferences, and in my master's thesis.

1. **Detecting suspicious activities in waste transport data via temporal and network-aware change detection** (ECML PKDD 2025 Workshops Track)
   Combines statistical change-detection tests with an LSTM-VAE deep learning model to flag abrupt or unusual shifts in company activity and connectivity within Portugal's waste transport network, validated against companies confirmed to be fraudulent by the regulators.
   [Paper](https://link.springer.com/chapter/10.1007/978-3-032-19096-3_1)

2. **Conditional Motif-based Graph Convolutional Network for Anomaly Detection in the Waste Management Network** (IDA 2026)
   Introduces a graph neural network that embeds triangular motif structures directly into its message-passing mechanism, improving detection of higher-order connectivity patterns over standard graph-based approaches.
   [Paper](https://link.springer.com/chapter/10.1007/978-3-032-23833-7_24)

3. **Detecting Suspicious Activities in Waste Transport Data via Temporal and Network-Aware Change Detection** (Master's thesis)
   Expands the anomaly detection framework developed for the ECML paper (item 1) by presenting a more detailed data analysis and applying a wider combination of features and anomaly detection methods.
   [Paper](https://github.com/saratoninb9/Data-Science-Portfolio/blob/81e45cf04246270ca2fe03da75d187334d6a70c8/Thesis_final.pdf) · [Poster](https://github.com/saratoninb9/Data-Science-Portfolio/blob/81e45cf04246270ca2fe03da75d187334d6a70c8/Thesis_poster.pdf)

---

## Group Project: Analysis of the Top 250 Portuguese Companies
*Data Analysis, Master's course, FEP — with António Silva*

Analyzed the largest 250 Portuguese companies by turnover (SABI database, 2022) to understand their economic and financial structure and identify meaningful patterns across the group. Reduced 19 raw economic, financial, and activity variables to 10 meaningful financial ratios, then combined univariate and bivariate analysis with Principal Component Analysis and Cluster Analysis to segment the companies. The analysis identified three well-defined company profiles (Strong, Stable, and Frail) and found that around 80% of the top 250 are financially and economically healthy, with a small group (~5%) of exceptionally large firms (turnover between €2–15 billion), and showed that economic activity type has a statistically significant effect on financial structure.

**Techniques:** exploratory data analysis, PCA, cluster analysis, correlation analysis, hypothesis testing

[Report](https://github.com/saratoninb9/Data-Science-Portfolio/blob/6b4c799ccdb5e83465758d971d80d96de217cb7c/Top250PortugueseCompanies.pdf) · [Poster](https://github.com/saratoninb9/Data-Science-Portfolio/blob/6b4c799ccdb5e83465758d971d80d96de217cb7c/Ant%C3%B3nioSilva_SaraOliveira_poster.pdf)

---

## Group Project: Image Similarity Detection for Fashion Retail Products (Parfois)
*Data Mining II, Master's course, FEP — with António Silva, Inês Barbosa, Kristin Cabrera, and Olga Pavlova*

Built a similarity-detection system for Parfois, a Portuguese fast-fashion accessories retailer, to identify visually and categorically similar products from their catalog, supporting product recommendations, purchase/inventory planning, and avoiding overstock of near-duplicate items. Combined product image data with tabular product attributes (category, material, color, price, description) and sales history. On the image side, applied a semi-automated segmentation pipeline (using the Segment Anything Model) to remove background noise before extracting embeddings from a pre-trained ResNet50, validated with Grad-CAM. On the tabular side, trained a self-supervised XGBoost feature extractor on product attributes. Combined both embedding types via cosine similarity and geometric mean to produce a final similarity score, returning at least 4 similar products per item. Extended the model to a simple sales-forecasting use case, predicting a product's expected sales as a similarity-weighted average of its closest matches.

**Techniques:** image segmentation (Segment Anything Model), CNN feature extraction (ResNet50), Grad-CAM, self-supervised learning (XGBoost), cosine similarity, embedding fusion, sales forecasting

[Presentation](https://github.com/saratoninb9/Data-Science-Portfolio/blob/6b4c799ccdb5e83465758d971d80d96de217cb7c/Parfois%20Presentation_AntonioInesKristinOlgaSara.pdf)
