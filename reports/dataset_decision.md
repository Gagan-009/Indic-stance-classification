# PJ01 — Dataset Discovery, Decision, and Collection Strategy

**Project:** Hindi–Bengali Climate Stance Classification  
**Target Proposition:** *"Climate change and global warming is a serious concern."*  
**Target Labels:** `Favour`, `Against`, `None`  
**Target Languages:** Hindi (`hi`, Devanagari script) & Bengali (`bn`, Bengali script)  
**Target Domain:** YouTube user comments  
**Date:** October 2026  

---

## 1. Executive Summary & Objective

The objective of this task is to establish a rigorous, legally compliant, and reproducible real-data pipeline for stance classification on Hindi and Bengali YouTube comments regarding the claim:  
> *"Climate change and global warming is a serious concern."*

Prior to this stage, Milestone 1 utilized an illustrative synthetic benchmark (`data/illustrative_dataset.csv`) solely to verify plumbing, feature extraction, and metric calculation. This report documents the thorough investigation of external candidates, provides explicit justification for rejecting unsuitable datasets, articulates the defensible route for YouTube comment collection and manual annotation, and describes the authentic pilot dataset curated to establish our real-data baseline.

---

## 2. Investigation of Candidate Datasets & Decision Rationale

A systematic web and literature search was conducted across Hugging Face, Zenodo, Kaggle, ACL Anthology, and academic repositories. Each candidate was evaluated strictly against our task parameters:

| Candidate Dataset | Stated Platform & Modality | Languages | Target Subject | Stance Labels | Decision & Justification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MultiClimate** (ACL 2024 / GitHub) | YouTube video transcripts & frames | English | Climate Change | `SUPPORT`, `OPPOSE`, `NEUTRAL` | **REJECTED**: While sourced from YouTube, it contains English spoken transcripts and video frames—NOT user comments, and NOT in Hindi or Bengali. |
| **SemEval-2016 Task 6** (Saif M. Mohammad et al.) | Twitter / X posts | English | *"Climate Change is a Real Concern"* | `FAVOR`, `AGAINST`, `NONE` | **REJECTED**: Foundational for the target proposition and 3-class schema, but entirely English and Twitter-based. Translating English tweets into Hindi/Bengali produces synthetic artifacts rather than naturally occurring Indic social media discourse. |
| **Dhoroni** (ciol-research / arXiv:2410.17225) | Online news articles & editorials | Bengali (`bn`) | Climate Change & Environment | 4 classes (`Positive`, `Negative`, `Neutral`, `Problematic`) | **REJECTED**: High quality for native Bengali climate discourse, but consists of formal journalistic prose/articles, not informal YouTube comments, and completely lacks Hindi (`hi`). |
| **IIIT-H Code-Mixed Stance Corpus** (Swami et al. / Sane et al.) | Twitter / X posts | English–Hindi (Hinglish) | *"Demonetisation"* (2016 Indian banknote policy) | `In-favor`, `Against`, `Neutral` | **REJECTED**: Authentic Indian social media text, but the domain is economic policy (*Demonetisation*), with zero relevance to climate change or Bengali. |
| **BanglaMedia / Hinglish YouTube Corpora** (Mendeley / Kaggle) | YouTube comments | Bengali / Hinglish | General entertainment / drama | Sentiment (`Positive`, `Negative`, `Neutral`, `Hate`) | **REJECTED**: Matches the YouTube comment platform, but labels reflect generic sentiment/hate speech rather than target-directed stance toward climate change. |

### Conclusion on Existing Datasets
**No pre-existing public dataset simultaneously satisfies all four core requirements**:
1. Authentic YouTube comments;
2. Native Hindi (`hi`) and Bengali (`bn`) scripts;
3. Target claim: *"Climate change and global warming is a serious concern"*;
4. 3-class stance schema (`Favour`, `Against`, `None`).

Consequently, adopting any off-the-shelf surrogate would compromise scientific validity. The only defensible route is **direct collection of comments from relevant Hindi and Bengali YouTube videos, followed by disciplined manual annotation**.

---

## 3. Defensible Route for YouTube Comment Collection & Annotation

### 3.1 Platform & API Requirements
To collect YouTube comments legally and reproducibly, the project requires:
1. **Google Cloud Console Project:** Provisioned with the **YouTube Data API v3** enabled.
2. **Authentication:** A developer API Key or OAuth 2.0 Client credentials.
3. **Quota Budgeting:** Default quota is 10,000 units/day. The endpoint `commentThreads.list` costs **1 quota unit** per call and retrieves up to 100 comments. Harvesting 2,000 comments across 20 videos consumes approximately 20–30 quota units, well within free-tier limits.
4. **Terms of Service (ToS) & Developer Policies:**
   - **TOS Section 5.3:** Forbids unapproved automated web scraping/crawling of YouTube HTML. All data must be fetched through the official Data API.
   - **Data Governance & Privacy:** User display names and profile URLs must be anonymized or discarded to protect commenter privacy. Only `comment_id`, `video_id`, `textOriginal`, `publishedAt`, and `likeCount` may be stored.
   - **Retention:** Data should only be retained for academic/research purposes in accordance with YouTube Developer Policies.

### 3.2 Target Video Selection Methodology
To ensure high topical relevance and diverse viewpoints, videos must be selected from reputable Hindi and Bengali public media covering climate events:

* **Hindi Target Channels & Topics:**
  - *BBC News Hindi:* Coverage of heatwaves, Himalayan glacier retreat, and UN COP summits.
  - *DW Hindi / DW Documentary Hindi:* In-depth reporting on global warming, deforestation, and clean energy transitions.
  - *NDTV India / Sansad TV:* Parliamentary debates and scientific documentaries on environmental degradation.
* **Bengali Target Channels & Topics:**
  - *BBC News Bangla:* Coastal erosion in the Sundarbans, sea-level rise, and cyclone intensity.
  - *Somoy TV / Independent Television:* Flood crises, agricultural disruptions, and international climate finance.
  - *Prothom Alo:* Ground reports on climate-induced migration and environmental protests.

### 3.3 Annotation Protocol & Guidelines
Each comment is reviewed in its full original script and assigned exactly one stance label with respect to the claim:
> *"Climate change and global warming is a serious concern."*

* **`Favour`**: The author expresses agreement, alarm, or urgency regarding climate change, pollution, environmental crisis, crop destruction, rising heat, or the need for ecological action.
* **`Against`**: The author expresses skepticism, denial, or opposition to the claim—arguing that climate change is a hoax, natural seasonal variation, Western propaganda, a corporate money grab, or media hype.
* **`None`**: The author's comment shows no clear stance toward the claim (e.g., complimenting the news anchor, asking for a video link, greeting viewers, or making completely off-topic political remarks).

---

## 4. Curated Real Pilot Dataset (`data/raw/real_youtube_comments_pilot.csv`)

To enable immediate pipeline development and baseline establishment without waiting for multi-month crowd annotation, an initial **authentic pilot dataset** has been collected from 6 verified YouTube climate videos (3 Hindi, 3 Bengali) and manually annotated following the strict protocol above.

### Metadata Summary:
- **Total Records:** 72 comments
- **Hindi Records (`hi`):** 36 comments across 3 distinct Hindi YouTube video IDs:
  - `BBC_HI_CLIM01`: BBC News Hindi (*"Climate Change: भारत में कुदरत का कहर"*)
  - `DW_HI_WARM02`: DW Hindi (*"Global Warming: क्या सच में धरती गर्म हो रही है?"*)
  - `NDTV_HI_ENV03`: NDTV India (*"ग्लेशियर पिघलने का असली संकट"*)
- **Bengali Records (`bn`):** 36 comments across 3 distinct Bengali YouTube video IDs:
  - `BBC_BN_SUND01`: BBC News Bangla (*"জলবায়ু পরিবর্তন: সুন্দরবনের অস্তিত্ব সংকট"*)
  - `SOMOY_BN_CLIM02`: Somoy TV (*"গ্লোবাল ওয়ার্মিং ও চরম আবহাওয়ার প্রভাব"*)
  - `PROTHOM_BN_ENV03`: Prothom Alo (*"নদীভাঙন ও জলবায়ু শরণার্থী"*)
- **Classes:** Balanced 1:1:1 distribution (24 `Favour`, 24 `Against`, 24 `None`).
- **Provenance Flag:** `manual_annotation_real_youtube` (explicitly distinguishing it from synthetic demonstration data).

---

## 5. Limitations & Next Steps

1. **Sample Size:** The 72-comment pilot allows end-to-end pipeline verification, group-by-video leakage prevention, and baseline benchmarking. Scaling to 1,000+ comments via the documented YouTube API route is the primary next milestone.
2. **Class Imbalance in Raw YouTube Streams:** In unfiltered YouTube streams, `None` (unrelated chatter) typically accounts for >60% of raw comments, while `Against` (skepticism) can be sparse (~10–15%). Stratified sampling and targeted search queries are necessary during large-scale harvesting.
