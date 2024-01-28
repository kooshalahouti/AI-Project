curacy(classifier):
#     eval_data_path = "eval_data.csv"
#     eval_data = load_data(eval_data_path)

#     cnt = 0
#     total = len(eval_data)

#     for features, true_label in eval_data:
#         predicted = classifier.classify(features)
#         if predicted == true_label:
#             cnt += 1

#     accuracy = cnt / total
#     return accuracy

# accuracy_on_eval_data = accuracy(nb_classifier)

# print(f"Accuracy on eval: {accuracy_on_eval_data:.2%}")