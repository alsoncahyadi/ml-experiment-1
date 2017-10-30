def evaluate_10fold(clf, feature, label):
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    
    number_of_split = 10
    
    kf = KFold(n_splits = number_of_split)
    print(kf)
    counter = 0 
    sum = 0    

    for train_index, test_index in kf.split(feature):

        clf.fit(feature[train_index], label[train_index])    
        label_predict = clf.predict(feature[test_index])
        
        counter += 1
        print("Accuracy " + "(fold " + str(counter) + ") : " + str(accuracy_score(label_predict, label[test_index])))
        sum += accuracy_score(label_predict, label[test_index])
        
    print("Average Accuracy : " + str(sum/number_of_split))
    
    return sum/number_of_split
