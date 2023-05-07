from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, \
    precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.tree import plot_tree
import numpy as np

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

    return roc_auc, 0

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

def compare_accuracies(model_names:list, acc_with_args:list, acc_without_args:list):
    # Define the models and their corresponding accuracies
    model_names = ["Log Regres", "Decision Tree", "Gradient Boosting", "Random Forest"]
    # Set the width of the bars
    bar_width = 0.06 * len(model_names)

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(model_names))
    r2 = [x + bar_width for x in r1]

    # Plot the bars
    plt.bar(r1, acc_with_args, color='blue', width=bar_width, label='With best hyperparameters')
    plt.bar(r2, acc_without_args, color='orange', width=bar_width, label='With default hyperparameters')
    # Add x-axis labels and title
    plt.xlabel('Models')
    plt.xticks([r + bar_width / 2 for r in range(len(model_names))], model_names)
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Models with and without best hyperparameters')

    # Add a legend
    plt.legend(loc='lower right')

    # Show the plot
    plt.show()

def plot_good_roc(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    # Calculate the FPR, TPR, and thresholds using the roc_curve function
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Plot the ROC curve
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--', color='gray')  # Plot the random guessing line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    # Find the optimal threshold that maximizes the Youden's J statistic (TPR-FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Optimal threshold:", optimal_threshold)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro')  # Plot the optimal threshold point
    plt.show()

    J = tpr - fpr
    optimal_threshold = thresholds[np.argmax(J)]

    y_pred = model.predict(X_test)

    accuracy_no_threshold = accuracy_score(y_test, y_pred)
    precision_no_threshold = precision_score(y_test, y_pred)
    recall_no_threshold = recall_score(y_test, y_pred)
    f1_no_threshold = f1_score(y_test, y_pred)

    fpr_no_threshold, tpr_no_threshold, thresholds_no_threshold = roc_curve(y_test, y_pred_prob)
    roc_auc_no_threshold = auc(fpr_no_threshold, tpr_no_threshold)

    # Adjust decision threshold based on optimal threshold
    y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)

    # Calculate performance metrics using adjusted threshold
    accuracy_optimal_threshold = accuracy_score(y_test, y_pred_optimal)
    precision_optimal_threshold = precision_score(y_test, y_pred_optimal)
    recall_optimal_threshold = recall_score(y_test, y_pred_optimal)
    f1_optimal_threshold = f1_score(y_test, y_pred_optimal)

    # Print results
    print("Performance metrics without optimal threshold:")
    print("Accuracy: {:.2f}".format(accuracy_no_threshold))
    print("Precision: {:.2f}".format(precision_no_threshold))
    print("Recall: {:.2f}".format(recall_no_threshold))
    print("F1 Score: {:.2f}".format(f1_no_threshold))
    print("AUC-ROC score without optimal threshold: {:.2f}".format(roc_auc_no_threshold))

    print("Performance metrics with optimal threshold:")
    print("Accuracy: {:.2f}".format(accuracy_optimal_threshold))
    print("Precision: {:.2f}".format(precision_optimal_threshold))
    print("Recall: {:.2f}".format(recall_optimal_threshold))
    print("F1 Score: {:.2f}".format(f1_optimal_threshold))
