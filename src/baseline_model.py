import numpy as np
import scipy.sparse as sparse
import joblib
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_prevectorized_model():
    print("Loading pre-computed TF-IDF sparse matrices...")
    X_train = sparse.load_npz('News_dataset/vectorized/X_train_tfidf.npz')
    X_test  = sparse.load_npz('News_dataset/vectorized/X_test_tfidf.npz')
    y_train = np.load('News_dataset/vectorized/y_train.npy')
    y_test  = np.load('News_dataset/vectorized/y_test.npy')

    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # FIX: Use PassiveAggressiveClassifier properly — the previous code was
    # trying to replicate it via SGDClassifier with incompatible args.
    # PAC is purpose-built for online text classification and hits 98%+ on ISOT.
    #
    # CalibratedClassifierCV wraps it so .predict_proba() works — this is what
    # the verification engine needs to output real confidence scores instead of
    # always saying "98%".
    print("Training PassiveAggressiveClassifier with isotonic calibration...")
    base_clf = PassiveAggressiveClassifier(
        C=0.1,              # regularization — tighter = less overfit on ISOT
        max_iter=200,
        tol=1e-4,
        random_state=42,
        class_weight='balanced',
    )

    # cv=3 does 3-fold cross-val internally for calibration — needed so the
    # classifier and calibrator don't train on the same data
    model = CalibratedClassifierCV(base_clf, method='isotonic', cv=3)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds) * 100

    print(f"\nAccuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['FAKE', 'REAL']))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    print("             Pred FAKE  Pred REAL")
    cm = confusion_matrix(y_test, preds)
    print(f"  Actual FAKE   {cm[0][0]:>6}     {cm[0][1]:>6}")
    print(f"  Actual REAL   {cm[1][0]:>6}     {cm[1][1]:>6}")

    joblib.dump(model, 'models/baseline_classifier.pkl')
    print("\nModel saved to models/baseline_classifier.pkl")

    return acc


if __name__ == "__main__":
    train_prevectorized_model()