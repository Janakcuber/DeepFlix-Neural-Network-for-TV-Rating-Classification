# DeepFlix-Neural-Network-for-TV-Rating-Classification

# 🎬 DeepFlix: Netflix Rating Classifier

Ever wonder how Netflix knows what's for kids and what's for adults? This project is my attempt to build a simple classifier using neural network to do just that!

This is a **Keras/TensorFlow** project that trains a neural network to predict a show's content rating. I have implemented the full ML pipeline by handling messy data to building a neural network.

## The Goal

The main challenge with the [Netflix dataset on Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows) is that the `rating` column is a mess. It has 14 different (and very imbalanced) categories, like 'TV-MA', 'PG-13', 'TV-Y7', and even bad data like '74 min'.

The goal wasn't just to build a model, but to **clean and engineer this messy data** into something a model could actually learn from.

## What This Project Does

* **Data Cleaning:** First, I filtered out all the bad 'min' values and nulls from the `rating` column.
* **Target Engineering (The Cool Part!):** Instead of 14 categories, I combined them into 3 simple, logical groups: `Kids`, `Teens`, and `Adult`. This made the problem *much* more solvable.
* **Feature Engineering:** Created new features from the data, like `age_of_show` (to see if older shows are different) and `duration_num` (to get a number from "90 min" or "2 Seasons").
* **Model Building:** Built a Sequential neural network using Keras (`TensorFlow`) with `Dense` and `Dropout` layers.
* **Smart Training:** Used `EarlyStopping` to prevent overfitting and `stratify` in the test split to make sure the 'Kids', 'Teens', and 'Adult' classes were fairly represented in the training data.
* **Evaluation:** I didn't just look at accuracy. I generated a `classification_report` and `confusion_matrix` to diagnose *exactly* where the model was getting confused.

## How It Works: The Workflow

1.  **Load & Explore:** Loaded `netflix.csv` and saw the `rating` column was a problem.
2.  **Clean & Engineer Target:** This was the most important step. Grouping the 14 ratings into 3 made the model *way* better (accuracy jumped from ~51% to 63.8%).
3.  **Engineer Features:** Created new columns from `release_year`, `duration`, and `type`.
4.  **Preprocess:**
    * One-hot encoded categorical features like `genre` and `country`.
    * Scaled all numerical features using `StandardScaler` so the neural network could train properly.
5.  **Build & Train:** Built and compiled the Keras model, then trained it using `EarlyStopping` to find the best version.
6.  **Evaluate:** Checked the results.

## Results & Analysis

The final model achieved a **test accuracy of 63.8%**.

But the accuracy score doesn't tell the whole story. The `classification_report` and `confusion_matrix` showed something really interesting:

* **What it does well:** It's great at identifying 'Kids' content (79% precision).
* **Where it struggles:** It gets very **confused between 'Teens' and 'Adults'**.
    * It predicted 247 'Adult' shows as 'Teens'.
    * It predicted 287 'Teens' shows as 'Adult'.

**Why?** This makes perfect sense! My current features (like genre, duration, country) are too similar for both categories. An R-rated 'Adult' action movie and a PG-13 'Teens' action movie look almost identical to the model. It's missing the key information!

## Future Improvements

This was a great learning experience. The 63.8% is a good baseline, but to make this *really* good, I'd focus on features:

1.  **Use the Text Data!** The `description` column is the biggest untapped resource. Using **TF-IDF** on it would give the model the keywords (like "violence," "drug use," "romance") it needs to tell 'Teens' and 'Adults' apart. This would almost certainly fix the main confusion.
2.  **Better Categoricals:** The `country` feature created hundreds of columns. I'd group rare countries into an "Other" category to reduce noise and help the model generalize better.
3.  **Compare Models:** I'd compare this neural network's performance to a `RandomForestClassifier` to see which one performs better on this kind of tabular data.
