import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def prepare_training_data(df: pd.DataFrame):
    """Separates the target column from features and performs a stratified train/validation split."""
    y_train = df['price_category']
    X_train = df.drop(columns=['price_category'])

    # Stratified split to preserve class proportions
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.30, stratify=y_train, random_state=0)
    return X_tr, X_val, y_tr, y_val


def evaluate_models(X_tr, X_val, y_tr, y_val):
    """ Trains classifiers, evaluates them using macro F1-score,
        and returns the top model with its evaluation metrics."""
    clfs = {
        "Tree": DecisionTreeClassifier(max_depth= 4, class_weight="balanced", random_state=0),
        "RF" : RandomForestClassifier(max_depth= 5, class_weight= "balanced", random_state=0),
        "RF1000" : RandomForestClassifier(max_depth= 5, n_estimators = 1000, class_weight= "balanced", random_state=0),
        "Bagging" : BaggingClassifier(random_state=0),
        "Boosting": AdaBoostClassifier(random_state=0),
        "xGB" : GradientBoostingClassifier(random_state=0),
        "HGB_Classifier" : HistGradientBoostingClassifier(class_weight= "balanced", random_state=0)
    }
    # Stores: (best_name, best_f1, best_y_pred, best_clf_object)
    best = (None,0,None, None)
    for name, clf in clfs.items():
        # Train model and predict on validation set
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)

        # Calculate macro F1-score
        f1 = f1_score(y_val, y_pred, average='macro')

        # Track the best performing model
        if best[1] < f1:
            best = (name, f1, y_pred, clf)
    return best