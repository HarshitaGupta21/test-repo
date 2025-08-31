import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def load_data():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return digits, data

def split_data(data, digits):
    return train_test_split(data, digits.target, test_size=0.5, shuffle=False)

def train_classifier(X_train, y_train):
    classifier = svm.SVC(gamma=0.001)
    classifier.fit(X_train, y_train)
    return classifier

def evaluate_model(classifier, X_test, y_test):
    predicted = classifier.predict(X_test)
    print(f"Classification report:\n{metrics.classification_report(y_test, predicted)}\n")
    print(f"Confusion matrix:\n{metrics.confusion_matrix(y_test, predicted)}")
    return predicted

def plot_predictions(digits, X_test, predicted):
    images_and_predictions = list(zip(digits.images[len(digits.images) // 2:], predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis("off")
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title(f"Prediction: {prediction}")

    plt.show()
