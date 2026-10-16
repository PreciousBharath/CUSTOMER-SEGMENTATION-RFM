<div align="center">

# 🧩 SegmentIQ — Customer Segmentation with RFM & K-Means

**Turn raw order history into actionable customer segments — Champions, At Risk, Promising, Hibernating.**

![Python-3.10%2B](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Streamlit-App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) ![Plotly-Charts](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white) ![scikit--learn-KMeans](https://img.shields.io/badge/scikit--learn-KMeans-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white) ![Domain-Data%20Science](https://img.shields.io/badge/Domain-Data%20Science-00C9FF?style=for-the-badge&logo=databricks&logoColor=white)

`Data Science · Unsupervised Learning`

</div>

---

![CUSTOMER-SEGMENTATION-RFM](assets/screenshot.png)

## 📌 Problem Statement

Marketing teams waste budget sending the same campaign to every customer. Segmentation lets them target the right message to the right group.

## ✨ Key Features

- **RFM feature engineering** (Recency, Frequency, Monetary) from transaction logs
- Log-transform + standard scaling to handle heavy-tailed spend
- **Elbow + silhouette** analysis to choose *k* (interactive slider)
- Automatic, rule-based **segment naming** and recommended marketing action
- 3D RFM scatter, PCA projection, spend box-plots, customer vs revenue share
- Download every customer with their segment as CSV

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Transactions] --> B[RFM Table]
    B --> C[log1p + StandardScaler]
    C --> D[Elbow & Silhouette]
    C --> E[K-Means]
    E --> F[Segment Naming]
    F --> G[Profiles + Actions]
    E --> H[PCA 2D View]
```

## ⚙️ How It Works

1. ~20,000 orders from 3,000 customers are simulated with hidden behaviour patterns.
2. RFM table is built per customer, `log1p` transformed and standardised.
3. K-Means (k-means++, 20 inits) clusters customers; silhouette score measures separation.
4. Clusters are named by a value score (F + M − R) and a frequency × recency quadrant.

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML / Logic | pandas, numpy, scikit-learn |
| UI & Visualisation | Streamlit, Plotly |

## 📁 Project Structure

```
CUSTOMER-SEGMENTATION-RFM/
├── app.py             # Streamlit dashboard
├── segmentation.py    # Data simulation, RFM, k-search, clustering, naming
├── assets/              # screenshots
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

```bash
git clone https://github.com/PreciousBharath/CUSTOMER-SEGMENTATION-RFM.git
cd CUSTOMER-SEGMENTATION-RFM
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> The project ships with a realistic built-in demo dataset, so it runs out of the box with no downloads or API keys.

## 🗺️ Roadmap

- [ ] Customer Lifetime Value (BG/NBD) model
- [ ] DBSCAN / GMM comparison
- [ ] Push segments to email marketing tools

## 👨‍💻 Author

**Bharath** — Software Developer · Project Engineer · Technical Trainer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-connectbharath-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/connectbharath)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://itsbharathportfolio.netlify.app)
[![GitHub](https://img.shields.io/badge/GitHub-PreciousBharath-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PreciousBharath)

⭐ If this project helped you, please give it a star!

## 📄 License

Released under the [MIT License](LICENSE).
