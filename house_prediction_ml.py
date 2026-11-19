import io
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Step 1: Handling Unstructured Text (Feature Engineering)
# ---------------------------------------------------------
def extract_text_features(df: pd.DataFrame, text_col: str = "description") -> pd.DataFrame:
    """Extracts numeric signal flags from unstructured listing descriptions."""
    if text_col in df.columns:
        desc_lower = df[text_col].astype(str).str.lower()
        
        # Binary keyword flags for location/proximity signals
        df["is_prime_location"] = desc_lower.str.contains("prime|luxury|executive", regex=True).astype(int)
        df["is_near_office"] = desc_lower.str.contains("office|tech|business|corridor|financial", regex=True).astype(int)
        df["is_near_transit"] = desc_lower.str.contains("metro|transit|bus|highway", regex=True).astype(int)
        
        # Drop raw text column after feature extraction
        df.drop(columns=[text_col], inplace=True)
    return df

# ---------------------------------------------------------
# Step 2: Training and Pipeline Function
# ---------------------------------------------------------
def train_and_evaluate_housing_model(
    df: pd.DataFrame,
    target_col: str = "price",
    test_size: float = 0.2,
    random_state: int = 0,
) -> tuple[LinearRegression, float, list[str]]:
    data = df.copy()

    # 1. Extract unstructured text features
    data = extract_text_features(data, text_col="description")

    # 2. Filter out zero/invalid prices
    if target_col in data.columns:
        data = data[data[target_col] > 0]

    # 3. Drop Zillow model estimates/leaked columns
    cols_to_drop = [
        "zestimate", "rentZestimate", "zestimateMinus30", "restimateMinus30",
        "zestimateLowPercent", "zestimateHighPercent", "restimateLowPercent",
        "zestimate_history", "hideZestimate", "restimateHighPercent",
        "livingAreaValue", "yearBuilt"
    ]
    data.drop(columns=cols_to_drop, errors="ignore", inplace=True)

    # 4. One-hot encode categorical columns
    categorical_columns = [col for col in ["homeStatus", "homeType"] if col in data.columns]
    if categorical_columns:
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # 5. Extract numeric features
    feature_cols = [
        col for col in data.columns 
        if col != target_col and pd.api.types.is_numeric_dtype(data[col])
    ]

    X = data[feature_cols].fillna(data[feature_cols].median())
    y = data[target_col]

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 7. Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 8. Evaluation
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"Validation MAPE: {mape * 100:.2f}%")

    return model, mape, feature_cols

# ---------------------------------------------------------
# Step 3: Run Training & Evaluation on New Examples
# ---------------------------------------------------------
if __name__ == "__main__":
    train_csv_data = """price,bedrooms,bathrooms,livingArea,homeType,homeStatus,yearBuilt,zestimate,rentZestimate,livingAreaValue,description
450000,3,2,1650,SINGLE_FAMILY,FOR_SALE,1998,448000,2400,1650,"Charming home close to tech corridor and downtown, quiet neighborhood"
620000,4,3,2400,SINGLE_FAMILY,FOR_SALE,2010,615000,3100,2400,"Prime location, walking distance to corporate office park and top schools"
310000,2,1,950,CONDO,FOR_SALE,2005,305000,1850,950,"Modern condo in city center near transit and shopping"
275000,2,2,1100,TOWNHOUSE,PENDING,2001,270000,1700,1100,"Well-maintained townhouse, 10 min drive to business hub"
850000,5,4,3600,SINGLE_FAMILY,FOR_SALE,2018,842000,4200,3600,"Luxury residence situated in prime tech district with premium fittings"
385000,3,2,1450,SINGLE_FAMILY,SOLD,1985,380000,2100,1450,"Classic suburban home, close to university and medical center"
525000,3,2.5,2100,TOWNHOUSE,FOR_SALE,2015,520000,2800,2100,"Spacious townhouse near major highway interchange and corporate offices"
0,3,2,1500,SINGLE_FAMILY,FOR_SALE,1995,350000,2000,1500,"Invalid record with 0 price to test filtering logic"
740000,4,3.5,2900,SINGLE_FAMILY,FOR_SALE,2016,735000,3600,2900,"Executive home near primary financial district and golf course"
220000,1,1,720,CONDO,FOR_SALE,2008,218000,1400,720,"Cozy starter apartment close to metro station and downtown"
490000,3,2,1800,SINGLE_FAMILY,PENDING,2004,485000,2600,1800,"Family home situated near prime community park and elementary school"
680000,4,3,2650,SINGLE_FAMILY,FOR_SALE,2012,672000,3300,2650,"Spacious layout, close to downtown offices and express bus stops" """

    eval_csv_data = """price,bedrooms,bathrooms,livingArea,homeType,homeStatus,yearBuilt,zestimate,rentZestimate,livingAreaValue,description
465000,3,2,1700,SINGLE_FAMILY,FOR_SALE,2000,460000,2450,1700,"Beautiful home within walking distance of tech offices"
635000,4,3,2450,SINGLE_FAMILY,FOR_SALE,2011,630000,3150,2450,"Corner lot in prime school district, near downtown hub"
325000,2,1.5,1000,CONDO,FOR_SALE,2006,320000,1900,1000,"Downtown condo with easy commute to financial center" """

    # Load data
    train_df = pd.read_csv(io.StringIO(train_csv_data))
    eval_df = pd.read_csv(io.StringIO(eval_csv_data))

    # Train model
    model, val_mape, feature_columns = train_and_evaluate_housing_model(train_df)

    # Process evaluation dataset
    eval_data = extract_text_features(eval_df.copy(), text_col="description")
    eval_data = pd.get_dummies(eval_data, columns=["homeStatus", "homeType"], drop_first=True)
    
    # Align evaluation columns with training features (filling missing dummies with 0)
    for col in feature_columns:
        if col not in eval_data.columns:
            eval_data[col] = 0
    X_eval = eval_data[feature_columns]
    y_actual = eval_data["price"]

    # Predict and evaluate
    y_eval_pred = model.predict(X_eval)
    eval_mape = mean_absolute_percentage_error(y_actual, y_eval_pred)

    print("\n--- Out-of-Sample Evaluation Results ---")
    for actual, predicted in zip(y_actual, y_eval_pred):
        print(f"Actual Price: ${actual:,.2f} | Predicted Price: ${predicted:,.2f}")
    print(f"Out-of-Sample Test MAPE: {eval_mape * 100:.2f}%")