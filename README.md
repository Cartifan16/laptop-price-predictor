# PriceRight Laptop Price Estimator
A Streamlit web app that predicts a fair, competitive price (in SGD) for a laptop based on its specs. Built for an online laptop retailer to price new listings instantly instead of manually researching comparable laptops for every new model.

## What it does
- User will set a laptop specs at the sidebar
- Then click the estimate price to run the specs through the train random forest and they will get the predicted price.
- It will show the exact feature values used for that prediction so the estimate isn't a black box.
- It will also plot the predicted price change as he ram increase when the other specs are fixed.
- Included input validation and error handling.

## Model
- For the model its random forest regressor. Its chosen after comparing against linear regression, decision tree and gradient boosting then tuned with randomizedsearchcv.
- Its trained on 1,270+ real laptop listings after removing duplicates and its feature engineered for screen ppi, touchscreen/ips, cpi brand/speed, HDD/SSD/Hybrid/Flash storage, simplified GPU brand, OS split into 4 category.

## How a prediction is made
1. The specs entered in the sidebar are assembled into a single-row DataFrame.
2. That row is one-hot encoded the same way the training data was (`pd.get_dummies`).
3. `reindex()` aligns the encoded row's columns to match the model's training columns exactly, filling in any missing columns with 0.
4. The model predicts a price, which is clamped to a minimum of $0 before being shown.

## Files needed to run

- `app.py`  this Streamlit app
- `model.pkl`  the trained model + column list, must be in the same folder as `app.py`


## Notes / limitations
- The RAM sensitivity chart is a sanity check on the model's own behaviour, not a promise of real-world price movement.
- Predictions are estimates for pricing guidance, not guaranteed sale prices.
