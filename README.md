# Amazon Product Review Sentiment Analyzer

A three-class sentiment analysis project for Amazon product reviews. It compares Multinomial Naïve Bayes and Logistic Regression using TF-IDF features with an LSTM neural network. The Streamlit dashboard reports accuracy and macro F1, shows sentiment proportions and confusion matrices, and predicts sentiment for a review entered by the user.

## Dataset

Download **Amazon Fine Food Reviews** from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). The source is large, so first make a smaller balanced sample locally:

```bash
python prepare_dataset.py Reviews.csv
```

This creates `Amazon_Reviews_Sample.csv` with up to 15,000 reviews per sentiment class. Upload that sample CSV to the app. It expects these columns:

- `Score`: star rating from 1 to 5
- `Text`: review text

The app maps ratings 1-2 to Negative, 3 to Neutral, and 4-5 to Positive. Both the preparation script and app read the CSV in chunks to limit memory use.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the preparation script on `Reviews.csv`:

   ```bash
   python prepare_dataset.py Reviews.csv
   ```

5. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

6. Upload `Amazon_Reviews_Sample.csv` in the sidebar, choose the per-class sample size and LSTM epochs, then click **Train / load models**.

The project config and uploader widget allow files up to 300 MB. If deploying on Streamlit Community Cloud, commit the `.streamlit/config.toml` folder and redeploy so the setting is applied.

The first training run may take several minutes. The trained models are cached for the current file and settings. If deploying to Streamlit Community Cloud, upload the smaller sample CSV through the app; do not commit either dataset CSV to the repository.
