from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from sklearn.tree import plot_tree

def get_best(accuracy_map, out=True):
    sorted_accuracy_map = sorted(accuracy_map.items(), key=lambda x: x[1], reverse=True)
    if out:
        for i, (k, v) in enumerate(sorted_accuracy_map, start=1):
            print(f"{i}. {k} ({v})")
        print()
    return sorted_accuracy_map
def get_report(y_test, y_pred, out=True):
    accuracy_score_ = accuracy_score(y_test, y_pred)
    confusion_matrix_ = confusion_matrix(y_test, y_pred)
    classification_report_ = classification_report(y_test, y_pred)
    if out:
        print("Accuracy score:", accuracy_score_)
        print("Confusion matrix:\n", confusion_matrix_)
        print("Classification report:\n", classification_report_)
        print()
    return accuracy_score_, confusion_matrix_, classification_report_


def plot_confusion_matrix(y_test, y_pred):
    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

def plot_roc(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # Add a dashed diagonal line for comparison
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_decision_tree(clf, column_values):
    plt.figure(figsize=(20, 12), dpi=100)
    plot_tree(clf, feature_names=column_values, class_names=["no", "yes"], filled=True, fontsize=10, max_depth=3)
    plt.show()

def plot_feature_importance(model, features):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=model.feature_importances_, y=features, ax=ax, orient='h')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    plt.show()
